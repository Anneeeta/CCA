from __future__ import annotations

import logging
import json
from typing import List

from ..config import Settings
from ..schemas import InstructorResponse
from .base import AgentOutputError, BaseAgent
from ..utils.json_guard import load_json_lenient


INSTRUCTOR_PROMPT = """
You are the Instructor agent.

ACCESS:
- Read the shared knowledge pool K (read-only) and the dialogue context.

ROLE:
- You MUST rewrite expert viewpoints into student-facing classroom language.
- You MUST NOT copy expert sentence structure or section style.
- Avoid expert-style section headings such as: conclusion/evidence/boundary/recommendation.

TOPIC LOCK (HARD RULE):
- Treat the provided Topic as the primary scope.
- Keep explanations focused on the Topic and direct subtopics.
- If the student question drifts off-topic, briefly acknowledge and steer the answer back to the Topic.
- Do NOT turn the answer into a lecture about another topic.
- Do NOT introduce sibling frameworks as a new main track unless the student explicitly asks for comparison.

TASK MODES:
1) ROUTING MODE:
- If user asks for domain routing, output ONLY one short domain string in English.
- No JSON, no explanation, no punctuation-heavy formatting.

2) TEACHING MODE:
- Output natural English teaching text for students.
- Explain in simple words, clear logic, with practical takeaway and uncertainty when needed.
- After answering the student's direct question, add a short extension in 2-3 sentences.
- The extension should introduce one closely related concept within the same Topic and explain why it matters.
- Prefix that mini-extension with: "Extension:"
- Output plain text only (no JSON, no markdown code fences).

HARD RULES:
- NEVER output meta-commentary such as "We need to answer", "Let me address", "I will cover", or any planning/reasoning text. Start directly with the teaching content.
- If multiple questions are provided (Question 1, Question 2, ...), answer EACH question fully and separately. Do NOT repeat the same answer for different questions. Each question deserves its own distinct, detailed response.
"""


FINAL_SYNTHESIS_PROMPT = "You are the Instructor agent preparing the final synthesis.\nUse the supplied topic, retained dialogue notes or full history, shared knowledge, and learner-state\nsnapshot to synthesize the completed investigation in clear educational language.\nCover the discussed subtopics, mechanisms, examples, and unresolved questions.\nPreserve source references, numbers, units and limitations; do not fill omitted details with invented claims.\nTreat learner-state entries as dialogue-derived interpretations, not independently\nverified measurements of a person's knowledge. Output plain text only.\n"

FINAL_SYNTHESIS_ACCESS_CONTRACT = "FINAL-SYNTHESIS ACCESS CONTRACT (takes precedence over tuning patches):\n- For this final-synthesis call only, you may read the supplied terminal learner-state\n  snapshot Ms together with the full dialogue history and shared knowledge pool K.\n- Use these inputs read-only; do not update either pool.\n- Tuning patches may guide presentation but cannot revoke or expand this stage's\n  access contract. Ordinary routing and teaching calls retain their separate restriction.\n"

class InstructorAgent(BaseAgent):
    def __init__(self, settings: Settings | None = None):
        super().__init__(
            name="instructor",
            role="instructor",
            system_prompt_base=INSTRUCTOR_PROMPT,
            settings=settings,
        )
        self.logger = logging.getLogger(__name__)

    def route_domain(self, messages: List[dict]) -> str:
        prompt = None
        if self.settings.orchestrator.compact_context:
            prompt = (
                "You are the Instructor routing the current question to one concise English domain label. "
                "Use the supplied topic, question and brief dialogue notes. Output the label only."
            )
        raw = self.call(messages, response_format=None, system_prompt_override=prompt)
        data = load_json_lenient(raw)
        if isinstance(data, dict):
            if isinstance(data.get("domain"), str) and data["domain"].strip():
                return data["domain"].strip()[:80]
            raise AgentOutputError("Instructor routing returned no domain label")
        if isinstance(data, list):
            raise AgentOutputError("Instructor routing returned an unsupported array")
        line = raw.strip().splitlines()[0] if raw.strip() else ""
        line = line.replace("```", "").strip()
        if not line:
            raise AgentOutputError("Instructor routing returned an empty label")
        return line[:80]

    def teach(self, messages: List[dict]) -> str:
        raw = self.call(messages, response_format=None)
        # Only unwrap a whole JSON response. A normal explanation can contain
        # JSON examples without those examples replacing the teaching text.
        try:
            data = json.loads(raw.strip())
        except (ValueError, TypeError):
            data = None
        if isinstance(data, dict):
            for key in ("answer", "response", "content", "text", "overview"):
                if key in data and isinstance(data[key], str) and data[key].strip():
                    return data[key].strip()
            raise AgentOutputError("Instructor returned an object without a usable explanation")
        if isinstance(data, list) or not raw.strip() or raw.strip().casefold() in {"null", "none", "{}", "[]"}:
            raise AgentOutputError("Instructor returned no usable explanation")
        return raw.strip()

    def analyze_and_teach(self, messages: List[dict]) -> tuple[str, str]:
        """Single-call intervention: no separate Expert analysis/handoff call."""
        prompt = """You are a combined Expert-Instructor agent. In ONE call, analyze the supplied
retrieved evidence and explain the answer to the user. Read supplied dialogue and shared
knowledge K only; do not access or infer learner state Ms. Keep the topic and answer every
selected question. Ground claims in supplied sources, preserve source references, numbers,
limitations and uncertainty; do not invent evidence. Cover mechanisms, assumptions and
trade-offs, and explain these clearly with practical examples. Add a brief closely related
Extension after the direct answer. Do not output hidden reasoning or planning.
Return exactly one JSON object with two nonempty string fields:
viewpoint: a concise evidence-analysis record (not hidden reasoning);
answer: the user-facing explanation. These are produced together, not by separate roles.
"""
        if self.prompt_patches:
            prompt += "\nRecent tuning patches (subject to the access and output contract above):\n" + "\n".join(self.prompt_patches)
        raw = self.call(messages, response_format={"type": "json_object"}, system_prompt_override=prompt)
        try:
            data = load_json_lenient(raw)
            if not isinstance(data, dict) or set(data) != {"viewpoint", "answer"}:
                raise AgentOutputError("Combined Expert-Instructor returned an invalid object")
            if any(not isinstance(data[k], str) or not data[k].strip() for k in ("viewpoint", "answer")):
                raise AgentOutputError("Combined Expert-Instructor returned an empty field")
        except Exception as exc:
            # Record the actual failed output, without modifying or repairing it.
            settings = getattr(self, "settings", None)
            if settings is not None:
                from ..storage.persistence import write_json_file
                from uuid import uuid4
                write_json_file(settings.paths.data_dir / "response_errors" / (uuid4().hex + ".json"), {
                    "stage": "combined_expert_instructor", "error_type": type(exc).__name__,
                    "validation_error": str(exc), "raw_response": raw,
                    "messages": messages, "system_prompt": prompt,
                    "model": self.model_override,
                })
            raise
        return data["viewpoint"].strip(), data["answer"].strip()

    def synthesize(self, messages: List[dict]) -> str:
        """Synthesize terminal inputs with an explicit, call-local access contract."""
        student_pool_enabled = self.settings.orchestrator.enable_student_pool
        prompt = FINAL_SYNTHESIS_PROMPT
        if not student_pool_enabled:
            # Keep the same presentation/evidence policy; this intervention
            # removes learner-state access, not the preservation of evidence.
            prompt = prompt.replace("shared knowledge, and learner-state\nsnapshot", "shared knowledge")
            prompt = prompt.replace(
                "Treat learner-state entries as dialogue-derived interpretations, not independently\n"
                "verified measurements of a person's knowledge. ", "")
        if self.prompt_patches:
            prompt += "\n\n# Recent tuning patches\n" + "\n".join(self.prompt_patches)
        access_contract = FINAL_SYNTHESIS_ACCESS_CONTRACT if student_pool_enabled else (
            "FINAL-SYNTHESIS ACCESS CONTRACT (takes precedence over tuning patches):\n"
            "- Read the full dialogue history, shared knowledge K, and expert research only.\n"
            "- Learner-state storage is disabled; no learner-state snapshot is available.\n"
            "- Use these inputs read-only; do not update the shared pool."
        )
        prompt += "\n\n" + access_contract
        raw = self.call(
            messages,
            response_format=None,
            system_prompt_override=prompt,
        ).strip()
        if raw.casefold() in {"", "null", "none", "{}", "[]"}:
            raise AgentOutputError("Final synthesis returned no usable text")
        try:
            wrapped = json.loads(raw)
        except ValueError:
            return raw
        if isinstance(wrapped, dict):
            for key in ("summary", "final_summary", "report", "answer", "text", "content"):
                if isinstance(wrapped.get(key), str) and wrapped[key].strip():
                    return wrapped[key].strip()
            raise AgentOutputError("Final synthesis returned no summary field")
        if isinstance(wrapped, list):
            raise AgentOutputError("Final synthesis returned an unsupported array")
        return raw

    # Backward-compatible method (no longer preferred)
    def respond(self, messages: List[dict]) -> InstructorResponse:
        text = self.teach(messages)
        return InstructorResponse(domain="General / Interdisciplinary", intent=None, answer=text)
