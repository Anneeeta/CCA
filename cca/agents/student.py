from __future__ import annotations

import logging
import re
from typing import List

from ..config import Settings
from ..deepseek_client import is_provider_error
from ..schemas import CandidateQuestion, CandidateQuestionsResponse, StudentEntry, StudentMsUpdate
from .base import AgentOutputError, BaseAgent, require_update_fields
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
- Return ONLY new or changed entries, never an unchanged copy of the existing pool.
- Reuse the existing id when revising content or moving an entry between buckets.
- Use id=null only for a new entry; omitted entries remain stored.
- Use remove_ids only to explicitly retire superseded entries. Do not infer removal from an omitted context item.
- unknown_unknown entries must be questions/hypotheses only.

OUTPUT FOR MODE B (STRICT JSON ONLY):
{
  "known": [
    {"id": null, "content": "...", "type": "known", "round_id": 1, "rationale": "...", "priority": 1}
  ],
  "known_unknown": [
    {"id": null, "content": "...", "type": "known_unknown", "round_id": 1, "rationale": "...", "priority": 1}
  ],
  "conflict": [
    {"id": null, "content": "...", "type": "conflict", "round_id": 1, "rationale": "...", "priority": 1}
  ],
  "unknown_unknown": [
    {"id": null, "content": "...", "type": "unknown_unknown", "round_id": 1, "rationale": "...", "priority": 1}
  ],
  "remove_ids": []
}

GENERAL CONSTRAINTS:
- All natural language VALUES must be in English.
- JSON keys must be in English only.
- No extra text, no markdown fences, no duplicated keys, no nested JSON in values.
- Keep responses concise.
"""


def _history_only_student_prompt() -> str:
    """Keep the question policy while removing the learner-state maintenance mode."""
    question_prompt = STUDENT_SYSTEM_PROMPT.split("\nB) If asked to UPDATE YOUR Ms", 1)[0]
    question_prompt = question_prompt.replace(
        "- You only see your own knowledge pool Ms and the dialogue context.",
        "- Use the supplied topic, guidance, and dialogue context to plan questions.",
    ).replace(
        "- If Ms contains off-topic items, ignore them when generating next-round candidate questions.",
        "- If history contains off-topic items, ignore them when generating next-round candidate questions.",
    ).replace(
        "- if Ms has any conflict items, include >=1 conflict-resolution question",
        "- if the dialogue contains competing accounts, include >=1 conflict-resolution question",
    )
    general_rules = STUDENT_SYSTEM_PROMPT.split("GENERAL CONSTRAINTS:", 1)[1]
    return question_prompt + "\nGENERAL CONSTRAINTS:" + general_rules


STUDENT_NO_STATE_POOL_PROMPT = _history_only_student_prompt()
NO_STATE_POOL_ACCESS_CONTRACT = (
    "LEARNER-STATE ABLATION CONTRACT (takes precedence over tuning patches):\n"
    "The persistent learner-state pool is disabled. Use topic, user guidance, and dialogue history only.\n"
    "Do not request, read, write, or reconstruct a persistent learner-state pool.\n"
    "The shared pool K remains inaccessible to Student."
)


def ms_update_entry_contract(round_id: int) -> str:
    """The compact prompt must retain the same entry types as the storage schema."""
    return (
        "\nENTRY CONTRACT (required in every bucket): "
        "Each bucket is a JSON array of objects, never strings. "
        "id is an existing string or null for a new entry; content is nonempty English text. "
        "type MUST equal the enclosing bucket name: known, known_unknown, conflict, or unknown_unknown. "
        "Do not use semantic labels such as fact, concept, question, or hypothesis as type. "
        f"round_id is the integer {round_id}. rationale is English text or null. "
        "priority is an integer or null, NEVER a word such as high, medium, or low. "
        "remove_ids is an array of existing string ids; use [] when nothing is removed. "
        "Return only these five top-level keys, with [] for buckets that have no changes. "
        "Example of one new known entry (replace example content, do not copy it): "
        '{"known":[{"id":null,"content":"A point explained by Instructor.","type":"known",'
        f'"round_id":{round_id},"rationale":null,"priority":null}}],'
        '"known_unknown":[],"conflict":[],"unknown_unknown":[],"remove_ids":[]} '
        "Use the corresponding bucket name as type for entries in every other bucket."
    )


class StudentAgent(BaseAgent):
    def __init__(self, settings: Settings | None = None):
        super().__init__(
            name="student",
            role="student",
            system_prompt_base=STUDENT_SYSTEM_PROMPT,
            settings=settings,
        )
        if not self.settings.orchestrator.enable_student_pool:
            self.system_prompt_base = STUDENT_NO_STATE_POOL_PROMPT
        self.logger = logging.getLogger(__name__)

    def build_system_prompt(self) -> str:
        prompt = super().build_system_prompt()
        if not self.settings.orchestrator.enable_student_pool:
            prompt += "\n\n" + NO_STATE_POOL_ACCESS_CONTRACT
        return prompt

    def generate_candidates(self, messages: List[dict]) -> List[CandidateQuestion]:
        try:
            # Allow one repair turn for malformed JSON.
            prompt = self.build_system_prompt()
            if self.settings.orchestrator.enable_student_pool and not self.settings.orchestrator.student_candidate_state:
                prompt = STUDENT_NO_STATE_POOL_PROMPT
                if self.prompt_patches and not self.settings.orchestrator.compact_context:
                    prompt += "\nRecent tuning patches:\n" + "\n".join(self.prompt_patches)
            if self.settings.orchestrator.compact_context and self.settings.orchestrator.enable_student_pool:
                prompt = prompt.split("\nB) If asked to UPDATE YOUR Ms", 1)[0] + "\nAll text values must be English. Return candidate JSON only."
                if self.prompt_patches:
                    prompt += "\nRecent tuning patches:\n" + "\n".join(self.prompt_patches)
            resp = self.call_json(messages, CandidateQuestionsResponse, max_attempts=2,
                                  system_prompt_override=prompt)
            return resp.candidates
        except Exception as e:
            if is_provider_error(e):
                raise
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
        if not self.settings.orchestrator.enable_student_pool:
            raise RuntimeError("Learner-state updates are disabled for this ablation.")
        try:
            prompt = None
            if self.settings.orchestrator.compact_context:
                prompt = (
                    "You are Student maintaining dialogue-derived state Ms. Read only the supplied Instructor answer and Ms. "
                    "Return incremental changes in known, known_unknown, conflict, unknown_unknown plus remove_ids. "
                    "known means explained in this dialogue, not verified user mastery; known_unknown is an unresolved question; "
                    "conflict records competing accounts; unknown_unknown is a potential blind-spot question/hypothesis. "
                    "Reuse an existing id for revisions or category moves. Use id=null for new entries. "
                    "Do not repeat unchanged entries, merge merely similar claims, or remove omitted context entries. "
                    "Each entry has id, content, type, round_id, rationale, priority. Values must be English. Return JSON only."
                )
            # Listing field names alone lets the model invent semantic labels and
            # textual priorities that cannot satisfy StudentEntry. Keep these
            # constraints even when the rest of the context is compacted.
            prompt = (prompt if prompt is not None else self.build_system_prompt()) + ms_update_entry_contract(round_id)
            update = self.call_json(messages, StudentMsUpdate, max_attempts=2, system_prompt_override=prompt)
            return require_update_fields(update, {"known", "known_unknown", "conflict", "unknown_unknown", "remove_ids"})
        except Exception as e:
            if is_provider_error(e):
                raise
            self.logger.warning("Student Ms JSON failed, using fallback: %s", e)
            raw = self.last_raw or ""
            data = load_json_lenient(raw)
            if isinstance(data, dict):
                return self._coerce_update(data, round_id)
            raise AgentOutputError("Learner-state update could not be recovered") from None

    def _coerce_update(self, data: dict, round_id: int) -> StudentMsUpdate:
        allowed = {"known", "known_unknown", "conflict", "unknown_unknown", "remove_ids"}
        if not set(data) & allowed or set(data) - allowed:
            raise AgentOutputError("Unrecognized learner-state update fields")
        buckets = {}
        for bucket in ["known", "known_unknown", "conflict", "unknown_unknown"]:
            items = data.get(bucket, [])
            if not isinstance(items, list):
                raise AgentOutputError("Learner-state buckets must be lists")
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
                        raise AgentOutputError("Invalid learner-state entry; no entries were discarded") from None
                elif not isinstance(item, str):
                    raise AgentOutputError("Learner-state entries must be objects or text")
            buckets[bucket] = entries
        removed = data.get("remove_ids", [])
        if not isinstance(removed, list) or any(not isinstance(value, str) or not value.strip() for value in removed):
            raise AgentOutputError("Learner-state removals must be a list of nonempty ids")
        buckets["remove_ids"] = removed
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
            question_type = None
            if isinstance(item, str):
                question = item
            elif isinstance(item, dict):
                question = item.get("question") or item.get("content") or item.get("text")
                question_type = item.get("type")
                rationale_val = item.get("rationale")
                if isinstance(rationale_val, str):
                    rationale = rationale_val.strip() or None
            if not isinstance(question, str):
                continue
            question = question.strip()
            if not question:
                continue
            try:
                out.append(CandidateQuestion(question=question, type=question_type, rationale=rationale))
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
