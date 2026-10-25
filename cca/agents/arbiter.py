from __future__ import annotations

import logging
import json
from typing import List

from ..config import Settings
from ..deepseek_client import is_provider_error
from ..schemas import ArbiterPatchResponse, KnowledgePoolUpdate
from .base import AgentOutputError, BaseAgent, require_update_fields


ARBITER_PROMPT = """
You are the Arbiter agent.
- You may read both the shared pool K and the student pool Ms.
- Tasks:
  1) Extract atomic knowledge items (1-3 sentences) from V^(exp) and A^(ins),
  normalize language, auto-assign weak labels, merge/prune to control size, and detect conflicts (keep both sides but annotate conflict_tag).
  2) Produce per-agent tuning patches for the next round. Patches must NOT break access control (e.g., never instruct Student to read K or Instructor to read Ms).
  3) For tuning patches, preserve topic focus: never recommend cross-topic drift or replacing the current Topic with sibling frameworks unless user guidance explicitly requests comparison.
- When producing K updates, return JSON: {"items": [KnowledgeItem...], "remove_ids": [], "round_summary": {...}}.
- Return only new/revised items; reuse stable ids for revisions and explicitly list superseded ids in remove_ids.
- The public round_summary must use ONLY the supplied public questions, Expert analysis, Instructor answer and evidence.
- Keep private Ms interpretations out of the public summary. Preserve uncertainty, numbers, units, conditions, sources,
  unresolved questions, conflicts and corrections. Never add unsupported conclusions or invent source URLs.
- Summary fields: round_id, questions, key_points, unresolved_questions, conflicts, resolved_questions, evidence_refs.
- When producing patches, return JSON: {"patches": [{"agent": "...", "prompt_patch": "...", "param_patch": {...}, "rationale": "..."}]}.
- All natural language text must be in English. When returning JSON, keep keys in English and keep text values in English.
- K update hard limits:
  - items length MUST be <= 8
  - each content MUST be short (prefer 1 concise sentence)
  - output compact single-line JSON only
  - no markdown/code fences
"""


K_UPDATE_CONTRACT = """
CURRENT TASK: UPDATE SHARED KNOWLEDGE K ONLY. This task-specific output contract
takes precedence over the other task examples above. Do NOT return tuning patches.
Return only {"items":[...],"remove_ids":[],"round_summary":...}.
items has at most 8 NEW or REVISED atomic claims; retain uncertainties and both
sides of disagreements. Do not copy unchanged K or erase contradictory evidence.
Each item has: id (existing string for a revision, null for a new item), content
(nonempty English text), label (short English string), source_role (exactly
"instructor", "expert", or "arbiter"), round_id (current integer round, 0 for
bootstrap), source_urls (array of URLs actually provided), and conflict_tag.
conflict_tag MUST be null or an object with ALL THREE fields:
{"conflict_with":["existing-counterpart-id"],"conflict_type":"disagreement","note":"Explain the conflicting claims."}
NEVER put a string such as "contested", "uncertainty", or a descriptive label in
conflict_tag. Use null if no actual counterpart id is available; never invent one.
Still preserve the disagreement/uncertainty in the claim text and its label.
Do not generate created_at or updated_at; storage manages those timestamps.
remove_ids is an array of existing string ids explicitly superseded, or [].
round_summary is null or an object with round_id (integer), and questions,
key_points, unresolved_questions, conflicts, resolved_questions, evidence_refs
(arrays of strings). Its content must be public dialogue evidence, never Ms.
Return a single JSON object without prose, Markdown or a second JSON object.
"""

PATCH_CONTRACT = """
CURRENT TASK: GENERATE TUNING PATCHES ONLY. This task-specific output contract
takes precedence over the other task examples above. Do NOT return K items,
remove_ids, round_summary, or any shared-knowledge update in this response.
Return exactly one JSON object {"patches":[...]} and no other top-level keys.
Each patch has agent ("student", "instructor", "expert", or "arbiter"),
prompt_patch (English string or null), param_patch (object or null), and rationale
(English string or null). param_patch may contain temperature (nonnegative number),
top_p (number greater than 0 and at most 1), max_tokens (positive integer).
Preserve access controls and topic lock. An empty {"patches":[]} is valid when
there is no warranted change. No prose, Markdown or a second JSON object.
"""


def _repair_request(messages, schema):
    # A schema failure needs targeted format feedback, not an identical paid retry.
    return [*messages, {"role": "user", "content": (
        "The previous response did not match this task's JSON schema. "
        "Regenerate the same requested task, preserving its supported content; "
        "do not switch between knowledge updates and tuning patches. "
        "Return one valid JSON object matching this schema: "
        + json.dumps(schema.model_json_schema(), ensure_ascii=True, separators=(",", ":"))
    )}]


class ArbiterAgent(BaseAgent):
    def __init__(self, settings: Settings | None = None):
        super().__init__(
            name="arbiter",
            role="arbiter",
            system_prompt_base=ARBITER_PROMPT,
            settings=settings,
        )
        if not self.settings.orchestrator.enable_student_pool:
            self.system_prompt_base = ARBITER_PROMPT.replace(
                "- You may read both the shared pool K and the student pool Ms.",
                "- Read the shared pool K and the supplied round outputs. Learner-state storage is disabled.",
            )
        # Arbiter remains chat-only, but now supports per-agent model override.
        self.chat_model = self.model_override or self.settings.model.chat_model
        self.logger = logging.getLogger(__name__)

    def build_system_prompt(self) -> str:
        prompt = super().build_system_prompt()
        if not self.settings.orchestrator.enable_student_pool:
            prompt += (
                "\n\nLEARNER-STATE ABLATION CONTRACT (takes precedence over tuning patches):\n"
                "The learner-state pool is disabled. Maintain shared knowledge K and generate tuning patches "
                "from the supplied outputs and candidate scores. Do not request or restore learner-state storage."
            )
        return prompt

    def update_shared_pool(self, messages: List[dict]) -> KnowledgePoolUpdate:
        # Force chat-only mode (no deep-thinking model).
        prompt = self.build_system_prompt() + "\n\n" + K_UPDATE_CONTRACT
        try:
            update = self.call_json(messages, KnowledgePoolUpdate, model=self.chat_model, use_response_format=True,
                                    system_prompt_override=prompt)
            return self._cap_k_update(update)
        except Exception as e:
            if is_provider_error(e):
                raise
            self.logger.warning("Arbiter K update JSON failed in chat mode (%s); retrying without response_format.", e)
            try:
                update = self.call_json(
                    _repair_request(messages, KnowledgePoolUpdate),
                    KnowledgePoolUpdate,
                    model=self.chat_model,
                    use_response_format=False,
                    system_prompt_override=prompt,
                )
                return self._cap_k_update(update)
            except Exception as e2:
                if is_provider_error(e2):
                    raise
                self.logger.error("Arbiter K update failed after fallback: %s", e2)
                raise AgentOutputError("Shared-state update failed after format recovery") from None

    def generate_patch(self, messages: List[dict]) -> ArbiterPatchResponse:
        prompt = self.build_system_prompt() + "\n\n" + PATCH_CONTRACT
        try:
            return self.call_json(messages, ArbiterPatchResponse, model=self.chat_model, use_response_format=True,
                                  system_prompt_override=prompt)
        except Exception as e:
            if is_provider_error(e):
                raise
            self.logger.warning("Arbiter patch JSON failed in chat mode (%s); retrying without response_format.", e)
            try:
                return self.call_json(_repair_request(messages, ArbiterPatchResponse), ArbiterPatchResponse,
                                      model=self.chat_model, use_response_format=False, system_prompt_override=prompt)
            except Exception as e2:
                if is_provider_error(e2):
                    raise
                self.logger.error("Arbiter patch failed after fallback: %s", e2)
                raise AgentOutputError("Tuning patch stage failed after format recovery") from None

    def _cap_k_update(self, update: KnowledgePoolUpdate) -> KnowledgePoolUpdate:
        require_update_fields(update, {"items", "remove_ids"})
        if len(update.items) > 8:
            raise AgentOutputError("Shared-state update exceeds its eight-item contract; no facts were discarded")
        capped_items = update.items
        for item in capped_items:
            item.content = " ".join(item.content.split())
            # Keep complete facts in storage; a separate context view controls
            # prompt size without cutting off conditions or numerical values.
        return update.model_copy(update={"items": capped_items})
