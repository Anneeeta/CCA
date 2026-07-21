from __future__ import annotations

import logging
import re
from typing import List

from ..config import Settings
from ..schemas import CandidateQuestion, CandidateQuestionsResponse, StudentEntry, StudentMsUpdate
from .base import BaseAgent
from ..utils.json_guard import load_json_lenient


STUDENT_SYSTEM_PROMPT = """
You are the Student agent in a cooperative classroom.

ACCESS CONTROL:
- You only see your own knowledge pool Ms and the dialogue context.
- You must NOT reference, assume, or mention access to the shared pool K.

PRIMARY GOAL:
- Ask high-quality questions that advance learning.
- Crucially, discover "unknown unknowns": important blind spots you haven't realized to ask about yet.

TOPIC LOCK (HARD RULE):
- Treat the provided Topic as the immutable anchor for the entire run.
- Every candidate question MUST remain within that Topic or its direct subtopics.
- Do NOT switch to a different domain/topic just because prior dialogue mentioned it.
- If Ms contains off-topic items, ignore them when generating next-round candidate questions.
- Cross-topic references are allowed only as brief comparison, and must explicitly serve the original Topic.
- If a candidate could fit many topics and lacks topic-specific terminology, reject it.
- Do NOT center questions on sibling frameworks/paradigms unless the user explicitly asks for comparison.

DEFINITION (MUST FOLLOW):
- known_unknown: you know what you don't understand yet.
- unknown_unknown: hidden blind spots (unstated assumptions, missing boundary conditions, missing evidence/evaluation, alternative explanations, distribution shift/failure cases).
- unknown_unknown must ALWAYS be phrased as questions/hypotheses needing verification, NEVER as asserted facts.

TASK MODES:
A) If asked to GENERATE CANDIDATE QUESTIONS:
- Produce diverse candidates with types.
- MUST include:
  - >=1 known_unknown question (clarification)
  - >=1 unknown_unknown question (blind-spot probing)
  - if Ms has any conflict items, include >=1 conflict-resolution question
- Run an internal topic-relevance gate before output:
  - keep only candidates that are clearly about Topic-specific concepts
  - drop candidates mainly about adjacent but different frameworks
- Prefer breadth across subtopics:
  - avoid all candidates clustering around the same narrow sub-issue
  - target at least 3 distinct subtopics when possible
- SUBTOPIC DIVERSITY RULE (HARD):
  - Each round must explore a DIFFERENT subtopic from previous rounds.
  - Before generating candidates, identify what subtopic was covered in the prior round.
  - At least 3 of 5 candidates must come from a subtopic NOT yet discussed.
  - If the previous round explored subtopic X, the next round MUST NOT explore subtopic X again.
  - A "subtopic" is a distinct angle, mechanism, use-case, stakeholder group, or application domain within the topic.

UNKNOWN-UNKNOWN BLIND-SPOT SCAN (MANDATORY CHECKLIST):
Generate unknown_unknown questions from at least TWO of these angles:
1) Hidden assumptions / prerequisites
2) Boundary conditions / failure modes
3) Evidence & evaluation gaps (what experiment/metric would validate?)
4) Distribution shift / new setting generalization
5) Alternative mechanisms / confounders (within the same Topic)
6) Trade-offs / unintended consequences

OUTPUT FOR MODE A (STRICT JSON ONLY):
{
  "candidates": [
    {
      "question": "<English question>",
      "type": "known_unknown|conflict|unknown_unknown",
      "rationale": "<English, 1 sentence: why this question matters>"
    }
  ]
}

B) If asked to UPDATE YOUR Ms (SELF-SUMMARY) at end of round:
- Read the instructor answer only (A^(ins)) plus your existing Ms.
- Update Ms into 4 buckets: known, known_unknown, conflict, unknown_unknown.
- unknown_unknown entries must be questions/hypotheses only.

OUTPUT FOR MODE B (STRICT JSON ONLY):
{
  "known": [
    {"content": "...", "type": "known", "round_id": 1, "rationale": "...", "priority": 1}
  ],
  "known_unknown": [
    {"content": "...", "type": "known_unknown", "round_id": 1, "rationale": "...", "priority": 1}
  ],
  "conflict": [
    {"content": "...", "type": "conflict", "round_id": 1, "rationale": "...", "priority": 1}
  ],
  "unknown_unknown": [
    {"content": "...", "type": "unknown_unknown", "round_id": 1, "rationale": "...", "priority": 1}
  ]
}

GENERAL CONSTRAINTS:
- All natural language VALUES must be in English.
- JSON keys must be in English only.
- No extra text, no markdown fences, no duplicated keys, no nested JSON in values.
- Keep responses concise.
"""



class StudentAgent(BaseAgent):
    def __init__(self, settings: Settings | None = None):
        super().__init__(
            name="student",
            role="student",
            system_prompt_base=STUDENT_SYSTEM_PROMPT,
            settings=settings,
        )
        self.logger = logging.getLogger(__name__)

    def generate_candidates(self, messages: List[dict]) -> List[CandidateQuestion]:
        try:
            # Allow one repair turn for malformed JSON.
            resp = self.call_json(messages, CandidateQuestionsResponse, max_attempts=2)
            return resp.candidates
        except Exception as e:
            self.logger.warning("Student candidates JSON failed, using fallback: %s", e)
            raw = self.last_raw or ""
            data = load_json_lenient(raw)
            coerced = self._coerce_candidates(data)
            if coerced:
                return coerced
            # Last-resort fallback: extract question-like lines from raw text.
            text_fallback = self._extract_question_lines(raw)
            if text_fallback:
                return text_fallback
            return [
                CandidateQuestion(
                    question="Could you provide a concise overview of the topic with key mechanisms, practical trade-offs, and common failure modes?"
                )
            ]

    def update_ms(self, messages: List[dict], round_id: int) -> StudentMsUpdate:
        try:
            return self.call_json(messages, StudentMsUpdate)
        except Exception as e:
            self.logger.warning("Student Ms JSON failed, using fallback: %s", e)
            raw = self.last_raw or ""
            data = load_json_lenient(raw)
            if isinstance(data, dict):
                return self._coerce_update(data, round_id)
            return StudentMsUpdate()

    def _coerce_update(self, data: dict, round_id: int) -> StudentMsUpdate:
        buckets = {}
        for bucket in ["known", "known_unknown", "conflict", "unknown_unknown"]:
            items = data.get(bucket, []) or []
            entries = []
            for item in items:
                if isinstance(item, str):
                    entries.append(StudentEntry(type=bucket, content=item, round_id=round_id))
                    continue
                if isinstance(item, dict):
                    item.setdefault("type", bucket)
                    item.setdefault("round_id", round_id)
                    try:
                        entries.append(StudentEntry.model_validate(item))
                    except Exception:
                        continue
            buckets[bucket] = entries
        return StudentMsUpdate(**buckets)

    def _coerce_candidates(self, data: dict | None) -> List[CandidateQuestion]:
        if not isinstance(data, dict):
            return []
        raw_candidates = data.get("candidates")
        if raw_candidates is None:
            # Common loose keys returned by some providers
            raw_candidates = data.get("questions") or data.get("items") or []
        if not isinstance(raw_candidates, list):
            return []
        out: List[CandidateQuestion] = []
        limit = max(1, int(self.settings.orchestrator.num_candidates))
        for item in raw_candidates:
            question = None
            rationale = None
            if isinstance(item, str):
                question = item
            elif isinstance(item, dict):
                question = item.get("question") or item.get("content") or item.get("text")
                rationale_val = item.get("rationale")
                if isinstance(rationale_val, str):
                    rationale = rationale_val.strip() or None
            if not isinstance(question, str):
                continue
            question = question.strip()
            if not question:
                continue
            try:
                out.append(CandidateQuestion(question=question, rationale=rationale))
            except Exception:
                continue
            if len(out) >= limit:
                break
        return out

    def _extract_question_lines(self, raw: str) -> List[CandidateQuestion]:
        if not raw or not raw.strip():
            return []
        out: List[CandidateQuestion] = []
        limit = max(1, int(self.settings.orchestrator.num_candidates))
        for line in raw.splitlines():
            text = line.strip()
            if not text:
                continue
            text = re.sub(r"^\d+[\).\-\s]+", "", text).strip()
            text = text.lstrip("-* ").strip()
            if "?" not in text:
                continue
            try:
                out.append(CandidateQuestion(question=text))
            except Exception:
                continue
            if len(out) >= limit:
                break
        return out
