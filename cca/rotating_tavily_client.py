"""One-request Tavily transport backed by a persistent shared key budget."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .tavily_key_rotation import TavilyKeyPool, TavilyKeyPoolError
from .tavily_rate_limit import TavilySearchRequestError, pooled_search_post


class RotatingTavilyClient:
    """Share paced searches and reserve immediately before each HTTP attempt.

    The legacy SDK can supply its own session, redirects, or retry behavior.
    This transport shares bounded 429 retries with all pooled search workers,
    counts every actual POST, and never follows redirects.
    """

    SEARCH_URL = "https://api.tavily.com/search"

    def __init__(self, key_pool: TavilyKeyPool, api_base_url: str | None = None):
        if api_base_url and api_base_url.rstrip("/") not in {
            "https://api.tavily.com", self.SEARCH_URL,
        }:
            raise TavilyKeyPoolError("Pooled Tavily search requires the official API endpoint.")
        self.key_pool = key_pool

    @classmethod
    def from_environment(cls, api_base_url: str | None = None) -> RotatingTavilyClient | None:
        """Use the configured ordered pool, preferring it to legacy single keys."""
        raw = os.environ.get("TAVILY_API_KEYS", "").strip()
        if not raw:
            return None
        try:
            keys = json.loads(raw)
        except (TypeError, ValueError):
            raise TavilyKeyPoolError("TAVILY_API_KEYS must be a JSON array of strings.") from None
        if not isinstance(keys, list):
            raise TavilyKeyPoolError("TAVILY_API_KEYS must be a JSON array of strings.")
        try:
            limit = int(os.environ.get("TAVILY_KEY_CALL_LIMIT", "950"))
        except (TypeError, ValueError):
            raise TavilyKeyPoolError("TAVILY_KEY_CALL_LIMIT must be a positive integer.") from None
        usage_path = os.environ.get("TAVILY_USAGE_PATH") or (
            Path(__file__).resolve().parent.parent / "data" / "tavily_key_usage.sqlite3"
        )
        return cls(TavilyKeyPool(keys, usage_path, per_key_limit=limit), api_base_url)

    def search(self, query: str, max_results: int = 3, include_raw_content: bool = False) -> dict:
        payload = {
            "query": query,
            "max_results": max_results,
            "include_raw_content": include_raw_content,
            "search_depth": "basic",
            "auto_parameters": False,
        }
        response = pooled_search_post(self.key_pool, payload, timeout=60)
        try:
            data = response.json()
        except (TypeError, ValueError):
            raise TavilySearchRequestError("Tavily search returned invalid JSON.") from None
        if not isinstance(data, dict) or not isinstance(data.get("results", []), list):
            raise TavilySearchRequestError("Tavily search returned an invalid result structure.")
        return data
