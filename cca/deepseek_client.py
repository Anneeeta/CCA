from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI, OpenAIError
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


class DeepSeekClient:
    """
    Thin wrapper over an OpenAI-compatible chat-completions endpoint.
    Handles retries, timeouts, and redacted logging.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        api_key = self.settings.deepseek_api_key
        if not api_key:
            logger.warning("CCA_API_KEY is not set; API calls will fail until provided.")
        self.client = OpenAI(api_key=api_key, base_url=self.settings.base_url)
        # Token usage tracking
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_tokens: int = 0
        self.total_reasoning_tokens: int = 0
        self.call_count: int = 0

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> str:
        model_to_use = model or self.settings.model.chat_model
        timeout_to_use = timeout or self.settings.request_timeout
        max_attempts = max(1, int(self.settings.max_retries))
        retryer = Retrying(
            retry=retry_if_exception_type(OpenAIError),
            wait=wait_exponential(multiplier=1, min=1, max=20),
            stop=stop_after_attempt(max_attempts),
            reraise=True,
        )

        for attempt in retryer:
            with attempt:
                start = time.time()
                try:
                    resp = self.client.chat.completions.create(
                        model=model_to_use,
                        messages=messages,
                        temperature=temperature if temperature is not None else self.settings.model.temperature,
                        top_p=top_p if top_p is not None else self.settings.model.top_p,
                        max_tokens=max_tokens if max_tokens is not None else self.settings.model.max_tokens,
                        response_format=response_format,
                        timeout=timeout_to_use,
                    )
                except OpenAIError:
                    logger.exception(
                        "Chat completion failed (attempt %s/%s).",
                        attempt.retry_state.attempt_number,
                        max_attempts,
                    )
                    raise
                duration = time.time() - start
                logger.info("Model call model=%s tokens=%s time=%.2fs", model_to_use, resp.usage, duration)
                # Accumulate token usage
                self.call_count += 1
                if resp.usage:
                    self.total_prompt_tokens += resp.usage.prompt_tokens or 0
                    self.total_completion_tokens += resp.usage.completion_tokens or 0
                    self.total_tokens += resp.usage.total_tokens or 0
                    rd = getattr(resp.usage, "completion_tokens_details", None)
                    if rd:
                        self.total_reasoning_tokens += getattr(rd, "reasoning_tokens", 0) or 0
                msg = resp.choices[0].message
                content = msg.content
                # Retry empty final output; internal reasoning is not user-facing content.
                if isinstance(content, str) and content:
                    return content
                reasoning = getattr(msg, "reasoning_content", None)
                if isinstance(reasoning, str) and reasoning:
                    logger.warning(
                        "DeepSeek returned empty content with non-empty reasoning_content. "
                        "Retrying (reasoning_content is internal chain-of-thought, not usable output)."
                    )
                    raise OpenAIError("Empty content with reasoning_content fallback not allowed")
                if isinstance(content, str):
                    return content
                logger.warning(
                    "Model returned non-text content (type=%s); coercing to empty string.",
                    type(content).__name__,
                )
                return ""

        raise RuntimeError("Unexpected retry flow termination.")
