from __future__ import annotations

import logging
from typing import List

from ..config import Settings
from ..schemas import ExpertResponse
from ..utils.json_guard import load_json_lenient
from .base import BaseAgent


EXPERT_PROMPT = """
You are the Expert agent.

ACCESS CONTROL:
- You read the shared knowledge pool K (read-only), the instructor-provided domain, and the student's question.
- You do NOT access, mention, or infer the student pool Ms.

ROLE DIFFERENTIATION (IMPORTANT):
- Do NOT write a teaching script.
- Provide a research-style viewpoint in English, with enough detail for mechanism, assumptions, boundaries, and trade-offs.

TOPIC LOCK (HARD RULE):
- Treat the provided Topic as the fixed scope.
- Analyze only content that is directly relevant to that Topic.
- If the question is partially off-topic, answer only the part that can be grounded in the Topic and ignore the rest.
- Do NOT pivot to sibling frameworks as the main subject unless comparison is explicitly requested.

GROUNDING (when search results are provided):
- If retrieved search snippets are provided in the context, you MUST ground your viewpoint in them.
- Cite specific findings from the snippets when relevant.
- Use the retrieved information to enrich mechanism details, evidence, and boundary conditions.
- Do NOT fabricate claims that contradict the retrieved evidence.

OUTPUT (STRICT JSON ONLY; no extra text, no markdown fences):
Return exactly ONE JSON object with EXACT key:
{
  "viewpoint": "single-line string; use \\n when line breaks are needed"
}

VALID EXAMPLE (MUST FOLLOW SHAPE):
{"viewpoint":"Conclusion: Early-stage plans can look stable but fail under real constraints.\\nEvidence: Initial assumptions often ignore resource limits and stakeholder behavior.\\nBoundary: The risk is lower when assumptions are tested with realistic scenarios.\\nRecommendation: Add explicit assumption checks and contingency triggers before execution."}

CONSTRAINTS:
- All natural language VALUES must be in English.
- Keys must be exactly: viewpoint.
- Do NOT include JSON objects/arrays besides the top-level object.
- JSON string rule: the "viewpoint" value must be a single-line JSON string; use \\n for line breaks, do NOT emit raw newlines inside the string.
"""


class ExpertAgent(BaseAgent):
    def __init__(self, settings: Settings | None = None):
        super().__init__(
            name="expert",
            role="expert",
            system_prompt_base=EXPERT_PROMPT,
            settings=settings,
        )
        self.logger = logging.getLogger(__name__)

    def generate_queries(self, messages: List[dict], num_queries: int = 3) -> List[str]:
        """Generate search queries for retrieval based on the student question."""
        query_prompt = (
            f"Based on the following context, generate {num_queries} concise search queries "
            "that would help find relevant information to answer the student's question.\n"
            "Return ONLY a JSON array of strings, e.g. [\"query1\", \"query2\", \"query3\"].\n"
            "No extra text, no markdown fences.\n\n"
        )
        for msg in messages:
            if msg.get("role") == "user":
                query_prompt += msg["content"]
                break
        try:
            raw = self.call([{"role": "user", "content": query_prompt}])
            self.last_raw = raw
            data = load_json_lenient(raw)
            if isinstance(data, list):
                return [str(q).strip() for q in data if q][:num_queries]
            if isinstance(data, dict):
                for key in ("queries", "search_queries", "questions"):
                    if key in data and isinstance(data[key], list):
                        return [str(q).strip() for q in data[key] if q][:num_queries]
        except Exception as e:
            self.logger.warning("Expert query generation failed: %s", e)
        return []

    def respond(self, messages: List[dict]) -> ExpertResponse:
        try:
            return self.call_json(messages, ExpertResponse)
        except Exception as e:
            self.logger.warning("Expert JSON failed, using fallback: %s", e)
            raw = self.last_raw or self.call(messages, response_format=None)
            raw_text = raw if isinstance(raw, str) else ""
            data = load_json_lenient(raw)
            if isinstance(data, dict):
                if "viewpoint" not in data:
                    for key in ("answer", "response", "content", "text"):
                        if key in data:
                            data["viewpoint"] = data[key]
                            break
                if "viewpoint" not in data:
                    # Backward-compatible coercion for older structured outputs.
                    parts = []
                    for key in ("thesis", "formal"):
                        if key in data and isinstance(data[key], str) and data[key]:
                            parts.append(data[key])
                    for key in ("evidence", "boundaries", "open_issues"):
                        if isinstance(data.get(key), list):
                            parts.extend([str(x) for x in data[key] if x])
                    if parts:
                        data["viewpoint"] = "\\n".join(parts)
                if "viewpoint" in data and data["viewpoint"]:
                    return ExpertResponse.model_validate(data)
            if raw_text.strip():
                return ExpertResponse(viewpoint=raw_text.strip())
            return ExpertResponse(viewpoint="No expert viewpoint was returned in this round.")
