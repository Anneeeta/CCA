from __future__ import annotations

import logging
from threading import Lock, local
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from openai import OpenAI, OpenAIError
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from .config import Settings, get_settings
from .api_pacing import pace_mimo_request

logger = logging.getLogger(__name__)

SILICONFLOW_V4_FLASH_MODEL = "deepseek-ai/DeepSeek-V4-Flash"


class ProviderRoutingMismatchError(RuntimeError):
    """A pinned route returned missing or unexpected model/provider metadata."""
    halt_batch = True


class ProviderConfigurationError(ValueError):
    """A local provider setup error; stop the batch before wasting later jobs."""
    halt_batch = True


class IncompleteResponseError(RuntimeError):
    """A returned response explicitly ended before a complete answer."""


def is_fatal_api_error(exc: BaseException) -> bool:
    """Permanent provider failures must never become parsing fallbacks."""
    return isinstance(exc, (ProviderRoutingMismatchError, ProviderConfigurationError)) or isinstance(exc, OpenAIError) and getattr(exc, "status_code", None) in {
        400, 401, 402, 403, 404, 405, 410, 413, 422,
    }


def is_provider_error(exc: BaseException) -> bool:
    """The client already handled transport retries; do not parse stale output."""
    return isinstance(exc, (OpenAIError, ProviderConfigurationError, ProviderRoutingMismatchError, IncompleteResponseError))


def retryable_api_error(exc: BaseException) -> bool:
    """Retry transient provider failures; reject invalid or unfunded requests once."""
    if not isinstance(exc, OpenAIError):
        return False
    status = getattr(exc, "status_code", None)
    return status is None or status in {408, 409, 429} or (isinstance(status, int) and status >= 500)


def normalize_model_name(model: str) -> str:
    """Resolve existing DeepSeek V4 Flash aliases to SiliconFlow's model ID."""
    normalized = model.strip()
    alias = normalized.casefold().replace("-", "").replace("_", "")
    if alias in {
        "deepseekv4f", "deepseekv4flash", "deepseekai/deepseekv4f", "deepseekai/deepseekv4flash",
    }:
        return SILICONFLOW_V4_FLASH_MODEL
    return normalized


def resolve_model_route(settings: Settings, model: str) -> tuple[str, str, Optional[str], str]:
    """Resolve a route without constructing a client or making a request."""
    normalized = normalize_model_name(model)
    lower = normalized.casefold()
    legacy_host = (urlparse(settings.base_url).hostname or "").casefold()
    if lower in {"deepseek-flash", "deepseek-v4-pro"}:
        provider = "deepseek"
    elif lower.startswith(("deepseek/", "xiaomi/")):
        provider = "openrouter"
    elif lower.startswith("mimo-"):
        provider = "mimo"
    elif lower.startswith(("deepseek-ai/", "pro/deepseek-ai/")):
        provider = "siliconflow"
    elif legacy_host in {"api.xiaomimimo.com", "token-plan-sgp.xiaomimimo.com"}:
        provider = "mimo"
    elif legacy_host == "api.siliconflow.cn":
        provider = "siliconflow"
    elif legacy_host == "openrouter.ai":
        provider = "openrouter"
    else:
        provider = "legacy"
    routes = {
        "deepseek": ("https://api.deepseek.com/v1", settings.deepseek_official_api_key),
        "mimo": (settings.mimo_base_url, settings.mimo_api_key),
        "siliconflow": (settings.siliconflow_base_url, settings.siliconflow_api_key),
        "openrouter": (settings.openrouter_base_url, settings.openrouter_api_key),
        "legacy": (settings.base_url, settings.deepseek_api_key),
    }
    endpoint, key = routes[provider]
    return provider, endpoint, key, normalized


class DeepSeekClient:
    """
    OpenAI-compatible wrapper with separate MiMo and SiliconFlow routes.
    Handles retries, timeouts, and redacted logging.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        # Clients are lazy: an evaluator can use a different provider from chat_model.
        self._clients: dict[tuple[str, str], OpenAI] = {}
        # The judge shares one client across workers. Per-invocation response
        # usage belongs to the calling thread, including paid empty retries.
        self._call_usage = local()
        self._usage_lock = Lock()
        # Token usage tracking
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_tokens: int = 0
        self.total_reasoning_tokens: int = 0
        self.call_count: int = 0
        self.request_attempt_count: int = 0
        self.responses_missing_usage: int = 0
        self.total_cost_usd: float = 0.0
        self.observed_providers: set[str] = set()
        self.observed_models: set[str] = set()

    def _route(self, model: str) -> tuple[str, str, Optional[str], str]:
        return resolve_model_route(self.settings, model)

    def _get_client(self, provider: str, base_url: str, api_key: Optional[str]) -> OpenAI:
        if not api_key:
            key_name = {"deepseek": "DEEPSEEK_JUDGE_API_KEY/DEEPSEEK_API_KEY", "mimo": "MIMO_API_KEY", "siliconflow": "SILICONFLOW_API_KEY", "openrouter": "OPENROUTER_API_KEY"}.get(
                provider, "CCA_API_KEY/DEEPSEEK_API_KEY/GONGYI_API_KEY"
            )
            raise ProviderConfigurationError(f"{key_name} is required for the {provider} provider")
        cache_key = (provider, base_url)
        if cache_key not in self._clients:
            options: dict[str, Any] = {
                "api_key": api_key,
                "base_url": base_url,
                "timeout": self.settings.request_timeout,
                # The wrapper owns retries; avoid multiplying attempts in the SDK.
                "max_retries": 0,
            }
            if provider == "mimo":
                options["default_headers"] = {"api-key": api_key}
            self._clients[cache_key] = OpenAI(**options)
        return self._clients[cache_key]

    @property
    def client(self) -> OpenAI:
        """Backward-compatible access to the default model's SDK client."""
        provider, base_url, api_key, _ = self._route(self.settings.model.chat_model)
        return self._get_client(provider, base_url, api_key)

    @property
    def last_call_usage(self) -> list[dict[str, Any]]:
        """Copy of this thread's latest call response usage; no request data."""
        return [dict(event) for event in getattr(self._call_usage, "events", [])]

    def _capture_response_usage(self, provider: str, model: str, usage: Any) -> None:
        snapshot: dict[str, Any] = {"provider": provider, "model": model}
        for source, fields in (
            (usage, ("prompt_tokens", "completion_tokens", "total_tokens",
                     "prompt_cache_hit_tokens", "prompt_cache_miss_tokens")),
            (getattr(usage, "prompt_tokens_details", None), ("cached_tokens",)),
            (getattr(usage, "completion_tokens_details", None),
             ("reasoning_tokens", "accepted_prediction_tokens", "rejected_prediction_tokens")),
        ):
            for field in fields:
                value = getattr(source, field, None)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    snapshot[field] = value
        self._call_usage.events.append(snapshot)

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        # Reset before routing/credential validation, so an early failure cannot
        # inherit usage from a previous successful call in this worker.
        self._call_usage.events = []
        provider, base_url, api_key, model_to_use = self._route(model or self.settings.model.chat_model)
        client = self._get_client(provider, base_url, api_key)
        timeout_to_use = timeout or self.settings.request_timeout
        request: dict[str, Any] = {
            "model": model_to_use,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.settings.model.temperature,
            "top_p": top_p if top_p is not None else self.settings.model.top_p,
            "timeout": timeout_to_use,
        }
        token_cap = max_tokens if max_tokens is not None else self.settings.model.max_tokens
        request["max_completion_tokens" if provider == "mimo" else "max_tokens"] = token_cap
        if response_format is not None:
            request["response_format"] = response_format
        if provider in {"mimo", "deepseek"}:
            default_thinking = self.settings.mimo_enable_thinking if provider == "mimo" else self.settings.deepseek_enable_thinking
            thinking = default_thinking if enable_thinking is None else enable_thinking
            request["extra_body"] = {"thinking": {"type": "enabled" if thinking else "disabled"}}
            if thinking:
                # MiMo fixes these values in thinking mode, so omit ignored overrides.
                request.pop("temperature")
                request.pop("top_p")
            if reasoning_effort is not None:
                raise ValueError("reasoning_effort is supported only on the SiliconFlow route")
        elif provider == "siliconflow":
            thinking = self.settings.siliconflow_enable_thinking if enable_thinking is None else enable_thinking
            request["extra_body"] = {"enable_thinking": thinking}
            effort = reasoning_effort or self.settings.siliconflow_reasoning_effort
            if effort is not None:
                if effort not in {"high", "max"}:
                    raise ValueError("SiliconFlow reasoning_effort must be high or max")
                if thinking:
                    request["extra_body"]["reasoning_effort"] = effort
        elif provider == "openrouter":
            thinking = self.settings.openrouter_enable_thinking if enable_thinking is None else enable_thinking
            extra: dict[str, Any] = {"reasoning": {"enabled": thinking}}
            if reasoning_effort is not None:
                extra["reasoning"]["effort"] = reasoning_effort
            if self.settings.openrouter_provider_only:
                extra["provider"] = {"only": list(self.settings.openrouter_provider_only),
                                     "allow_fallbacks": False, "require_parameters": True}
            request["extra_body"] = extra
        elif enable_thinking is not None or reasoning_effort is not None:
            raise ValueError("Thinking options require a configured MiMo or SiliconFlow route")
        max_attempts = max(1, int(self.settings.max_retries))
        retryer = Retrying(
            retry=retry_if_exception(retryable_api_error),
            wait=wait_exponential(multiplier=1, min=1, max=20),
            stop=stop_after_attempt(max_attempts),
            reraise=True,
        )

        for attempt in retryer:
            with attempt:
                if provider == "mimo":
                    pace_mimo_request(base_url, model_to_use)
                start = time.time()
                with self._usage_lock:
                    self.request_attempt_count += 1
                try:
                    resp = client.chat.completions.create(**request)
                except OpenAIError as exc:
                    # Provider error bodies may echo request content; never log them.
                    logger.warning(
                        "Chat completion failed provider=%s (attempt %s/%s) error=%s status=%s.",
                        provider,
                        attempt.retry_state.attempt_number,
                        max_attempts,
                        type(exc).__name__,
                        getattr(exc, "status_code", None),
                    )
                    raise
                duration = time.time() - start
                self._capture_response_usage(provider, model_to_use, getattr(resp, "usage", None))
                # Account every returned response before checking content or
                # route metadata. Rejected/empty responses can still be billed.
                event = self._call_usage.events[-1]
                with self._usage_lock:
                    self.call_count += 1
                    self.responses_missing_usage += not all(key in event for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
                    self.total_prompt_tokens += event.get("prompt_tokens", 0)
                    self.total_completion_tokens += event.get("completion_tokens", 0)
                    self.total_tokens += event.get("total_tokens", 0)
                    self.total_reasoning_tokens += event.get("reasoning_tokens", 0)
                    reported_provider = getattr(resp, "provider", None)
                    self.observed_providers.add(reported_provider if isinstance(reported_provider, str) and reported_provider else provider)
                    actual_model = getattr(resp, "model", None)
                    if isinstance(actual_model, str):
                        self.observed_models.add(actual_model)
                        event["response_model"] = actual_model
                if provider == "openrouter":
                    cost = getattr(resp.usage, "cost", None)
                    if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost >= 0:
                        with self._usage_lock:
                            self.total_cost_usd += cost
                        self._call_usage.events[-1]["cost_usd"] = cost
                    actual_provider = getattr(resp, "provider", None)
                    actual_model = getattr(resp, "model", None)
                    if isinstance(actual_provider, str):
                        with self._usage_lock:
                            self.observed_providers.add(actual_provider)
                        self._call_usage.events[-1]["reported_provider"] = actual_provider
                    if isinstance(actual_model, str):
                        with self._usage_lock:
                            self.observed_models.add(actual_model)
                        self._call_usage.events[-1]["response_model"] = actual_model
                    allowed = {value.split("/")[0].lower() for value in self.settings.openrouter_provider_only}
                    if allowed and (not actual_provider or actual_provider.lower() not in allowed):
                        raise ProviderRoutingMismatchError("Pinned provider metadata did not match")
                    if model_to_use == "deepseek/deepseek-v4-flash" and actual_model not in {
                        model_to_use, "deepseek/deepseek-v4-flash-20260423",
                    }:
                        raise ProviderRoutingMismatchError("Preview model metadata did not match")
                logger.info("Chat call provider=%s model=%s tokens=%s time=%.2fs", provider, model_to_use, resp.usage, duration)
                if not getattr(resp, "choices", None):
                    raise OpenAIError("Provider returned no response choice")
                msg = resp.choices[0].message
                content = msg.content
                finish_reason = getattr(resp.choices[0], "finish_reason", None)
                if isinstance(finish_reason, str):
                    event["finish_reason"] = finish_reason
                    if finish_reason != "stop":
                        error = IncompleteResponseError("Provider did not return a completed answer")
                        error.partial_text = content if isinstance(content, str) else ""
                        error.finish_reason = finish_reason
                        raise error
                # reasoning model: content 为空时重试，不使用 reasoning_content
                if isinstance(content, str) and content.strip():
                    return content
                reasoning = getattr(msg, "reasoning_content", None)
                if isinstance(reasoning, str) and reasoning:
                    logger.warning(
                        "Model returned empty content with non-empty reasoning_content. "
                        "Retrying (reasoning_content is internal chain-of-thought, not usable output)."
                    )
                    raise OpenAIError("Empty content with reasoning_content fallback not allowed")
                raise OpenAIError("Provider returned empty or non-text content")

        raise RuntimeError("Unexpected retry flow termination.")
