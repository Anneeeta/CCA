from __future__ import annotations

import logging
from typing import Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

from ..config import Settings, get_settings
from ..deepseek_client import DeepSeekClient
from ..schemas import AgentPatch
from ..utils import json_guard

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class BaseAgent:
    """
    Base class for all agents.
    """

    def __init__(
        self,
        name: str,
        role: str,
        system_prompt_base: str,
        settings: Optional[Settings] = None,
    ):
        self.name = name
        self.role = role
        self.settings = settings or get_settings()
        self.client = DeepSeekClient(self.settings)
        self.system_prompt_base = system_prompt_base
        self.prompt_patches: List[str] = []
        self.model_override: Optional[str] = self.settings.model.get_agent_model(name)
        self.param_overrides: Dict[str, float | int] = self.settings.model.get_agent_param_overrides(name)
        self.last_raw: Optional[str] = None

    # Prompt management -----------------------------------------------------

    def build_system_prompt(self) -> str:
        prompt = self.system_prompt_base
        if self.prompt_patches:
            prompt += "\n\n# Recent tuning patches\n" + "\n".join(self.prompt_patches)
        return prompt

    def apply_patch(self, patch: AgentPatch) -> None:
        if patch.prompt_patch:
            self.prompt_patches.append(patch.prompt_patch.strip())
        if patch.param_patch:
            for key, value in patch.param_patch.model_dump(exclude_none=True).items():
                if key == "max_tokens":
                    continue  # The Arbiter cannot change token caps at runtime.
                self.param_overrides[key] = value
        if patch.rationale:
            logger.info("Applied patch to %s: %s", self.name, patch.rationale)

    # Core call methods -----------------------------------------------------

    def call(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        response_format: Optional[Dict[str, str]] = None,
    ) -> str:
        system_prompt = self.build_system_prompt()
        full_messages = [{"role": "system", "content": system_prompt}, *messages]
        params = self.param_overrides
        return self.client.chat_completion(
            messages=full_messages,
            model=model or self.model_override,
            temperature=params.get("temperature"),
            top_p=params.get("top_p"),
            max_tokens=params.get("max_tokens"),
            response_format=response_format,
        )

    def call_json(
        self,
        messages: List[Dict[str, str]],
        schema: Type[T],
        model: Optional[str] = None,
        max_attempts: int = 1,
        use_response_format: bool = True,
    ) -> T:
        """
        Request JSON output and validate against schema.
        """
        attempt_messages = list(messages)
        for attempt in range(1, max_attempts + 1):
            content = self.call(
                attempt_messages,
                model=model,
                response_format={"type": "json_object"} if use_response_format else None,
            )
            self.last_raw = content
            try:
                return json_guard.parse_with_schema(content, schema)
            except Exception as e:
                snippet = content[:500].replace("\n", " ")
                logger.warning("%s JSON parse attempt %s failed: %s | content[:500]=%s", self.name, attempt, e, snippet)
                if attempt == max_attempts:
                    raise
                # Ask model to repair JSON
                attempt_messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": (
                            f"Previous output had parsing issues: {e}. "
                            f"Return ONLY valid JSON for schema: {schema.model_json_schema()}"
                        ),
                    },
                ]
