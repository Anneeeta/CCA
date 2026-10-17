from __future__ import annotations

import logging
from typing import List

from ..config import Settings
from ..schemas import InstructorResponse
from .base import BaseAgent
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
        raw = self.call(messages, response_format=None)
        data = load_json_lenient(raw)
        if isinstance(data, dict) and data.get("domain"):
            return str(data["domain"]).strip()
        line = raw.strip().splitlines()[0] if raw.strip() else ""
        line = line.replace("```", "").strip()
        return line[:80] if line else "General / Interdisciplinary"

    def teach(self, messages: List[dict]) -> str:
        raw = self.call(messages, response_format=None)
        data = load_json_lenient(raw)
        if isinstance(data, dict):
            for key in ("answer", "response", "content", "text", "overview"):
                if key in data and isinstance(data[key], str) and data[key].strip():
                    return data[key].strip()
        return raw.strip()

    # Backward-compatible method (no longer preferred)
    def respond(self, messages: List[dict]) -> InstructorResponse:
        text = self.teach(messages)
        return InstructorResponse(domain="General / Interdisciplinary", intent=None, answer=text)
