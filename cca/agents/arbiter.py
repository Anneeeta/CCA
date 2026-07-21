from __future__ import annotations

import logging
from typing import List

from ..config import Settings
from ..schemas import ArbiterPatchResponse, KnowledgePoolUpdate
from .base import BaseAgent


ARBITER_PROMPT = """
You are the Arbiter agent.
- You may read both the shared pool K and the student pool Ms.
- Tasks:
  1) Extract atomic knowledge items (1-3 sentences) from V^(exp) and A^(ins), 
  normalize language, auto-assign weak labels, merge/prune to control size, and detect conflicts (keep both sides but annotate conflict_tag).
  2) Produce per-agent tuning patches for the next round. Patches must NOT break access control (e.g., never instruct Student to read K or Instructor to read Ms).
  3) For tuning patches, preserve topic focus: never recommend cross-topic drift or replacing the current Topic with sibling frameworks unless user guidance explicitly requests comparison.
- When producing K updates, return JSON: {"items": [KnowledgeItem...]}.
- When producing patches, return JSON: {"patches": [{"agent": "...", "prompt_patch": "...", "param_patch": {...}, "rationale": "..."}]}.
- All natural language text must be in English. When returning JSON, keep keys in English and keep text values in English.
- K update hard limits:
  - items length MUST be <= 8
  - each content MUST be short (prefer 1 concise sentence)
  - output compact single-line JSON only
  - no markdown/code fences
"""


class ArbiterAgent(BaseAgent):
    def __init__(self, settings: Settings | None = None):
        super().__init__(
            name="arbiter",
            role="arbiter",
            system_prompt_base=ARBITER_PROMPT,
            settings=settings,
        )
        # Arbiter remains chat-only, but now supports per-agent model override.
        self.chat_model = self.model_override or self.settings.model.chat_model
        self.logger = logging.getLogger(__name__)

    def update_shared_pool(self, messages: List[dict]) -> KnowledgePoolUpdate:
        # Force chat-only mode (no deep-thinking model).
        try:
            update = self.call_json(messages, KnowledgePoolUpdate, model=self.chat_model, use_response_format=True)
            return self._cap_k_update(update)
        except Exception as e:
            self.logger.warning("Arbiter K update JSON failed in chat mode (%s); retrying without response_format.", e)
            try:
                update = self.call_json(
                    messages,
                    KnowledgePoolUpdate,
                    model=self.chat_model,
                    use_response_format=False,
                )
                return self._cap_k_update(update)
            except Exception as e2:
                self.logger.error("Arbiter K update failed after fallback: %s", e2)
                return KnowledgePoolUpdate(items=[])

    def generate_patch(self, messages: List[dict]) -> ArbiterPatchResponse:
        try:
            return self.call_json(messages, ArbiterPatchResponse, model=self.chat_model, use_response_format=True)
        except Exception as e:
            self.logger.warning("Arbiter patch JSON failed in chat mode (%s); retrying without response_format.", e)
            try:
                return self.call_json(messages, ArbiterPatchResponse, model=self.chat_model, use_response_format=False)
            except Exception as e2:
                self.logger.error("Arbiter patch failed after fallback: %s", e2)
                return ArbiterPatchResponse(patches=[])

    def _cap_k_update(self, update: KnowledgePoolUpdate) -> KnowledgePoolUpdate:
        # Hard cap size to avoid long responses and truncation.
        capped_items = update.items[:8]
        for item in capped_items:
            item.content = " ".join(item.content.split())
            if len(item.content) > 220:
                item.content = item.content[:220].rstrip() + "..."
        return KnowledgePoolUpdate(items=capped_items)
