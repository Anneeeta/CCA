from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path
from typing import List, Optional
import requests

from .tavily_key_rotation import TavilyKeyPool, TavilyKeyPoolError
from .tavily_rate_limit import TavilySearchRequestError, pooled_search_post

logger = logging.getLogger(__name__)


@dataclass
class RetrievedSnippet:
    """A single retrieved search result."""
    title: str
    url: str
    snippet: str
    description: str = ""
    metadata: dict = field(default_factory=dict)


class RetrievalError(RuntimeError):
    """An enabled retrieval stage failed, rather than finding zero results."""


def interleave_snippets(groups):
    result, seen = [], set()
    for row in zip_longest(*groups):
        for snippet in row:
            if snippet is not None and snippet.url not in seen:
                result.append(snippet)
                seen.add(snippet.url)
    return result


class DuckDuckGoRetriever:
    """
    Lightweight DuckDuckGo web retriever for CCA.
    No API key required. Uses duckduckgo_search library.
    """

    def __init__(
        self,
        max_results: int = 5,
        safe_search: str = "On",
        region: str = "us-en",
        strict_errors: bool = False,
    ):
        self.max_results = max_results
        self.safe_search = safe_search
        self.region = region
        self.strict_errors = strict_errors
        self._ddgs = None

    def _ensure_ddgs(self):
        if self._ddgs is None:
            try:
                # 尝试新的包名 ddgs
                from ddgs import DDGS
                self._ddgs = DDGS()
            except ImportError:
                try:
                    # 回退到旧的包名
                    from duckduckgo_search import DDGS
                    self._ddgs = DDGS()
                except ImportError:
                    raise ImportError(
                        "ddgs or duckduckgo_search is required. Install with: pip install ddgs"
                    )

    def search(self, query: str) -> List[RetrievedSnippet]:
        """Search DuckDuckGo and return structured snippets."""
        self._ensure_ddgs()
        results: List[RetrievedSnippet] = []
        try:
            raw = self._ddgs.text(
                query,
                max_results=self.max_results,
            )
            for d in raw:
                if not isinstance(d, dict):
                    continue
                url = d.get("href", "")
                title = d.get("title", "")
                body = d.get("body", "")
                if url and title and body:
                    results.append(RetrievedSnippet(
                        title=title,
                        url=url,
                        snippet=body,
                        description=body,
                    ))
        except Exception as e:
            if self.strict_errors:
                raise RetrievalError(f"DuckDuckGo retrieval failed ({type(e).__name__})") from None
            logger.warning("DuckDuckGo search failed for query '%s': %s", query, e)
        return results

    def search_multiple(self, queries: List[str]) -> List[RetrievedSnippet]:
        """Search multiple queries and deduplicate by URL."""
        seen_urls = set()
        all_results: List[RetrievedSnippet] = []
        for query in queries:
            for snippet in self.search(query):
                if snippet.url not in seen_urls:
                    seen_urls.add(snippet.url)
                    all_results.append(snippet)
        return all_results

    @staticmethod
    def format_snippets(snippets: List[RetrievedSnippet], max_chars: int = 3000) -> str:
        """Format retrieved snippets into a readable context string."""
        from .memory.context_view import excerpt
        if max_chars <= 0:
            return ""
        if not snippets:
            return "(no search results)"[:max_chars]
        lines = []
        total = 0
        for i, s in enumerate(snippets, 1):
            separator = 2 if lines else 0
            header = f"[{i}] {s.title}\n    URL: {s.url}\n    "
            remaining = max_chars - total - separator
            # Share the remaining budget, so one long first passage cannot
            # silently consume the evidence space for every later question.
            remaining //= len(snippets) - i + 1
            if len(header) >= remaining:
                continue
            entry = header + excerpt(s.snippet, remaining - len(header))
            lines.append(entry)
            total += separator + len(entry)
        return "\n\n".join(lines)


class TavilyRetriever:
    """
    Tavily web retriever for CCA.
    Supports both official Tavily API and public welfare API.
    """

    # Official Tavily endpoint by default; custom endpoints require explicit configuration.
    DEFAULT_API_URL = "https://api.tavily.com/search"

    OFFICIAL_API_URL = "https://api.tavily.com/search"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        max_results: int = 5,
        search_depth: str = "basic",
        *,
        api_keys: Optional[List[str]] = None,
        usage_path: Optional[Path | str] = None,
        per_key_limit: int = 950,
        strict_errors: bool = False,
    ):
        self.api_key = api_key
        self.strict_errors = strict_errors
        self._key_pool = None
        if api_keys:
            self._key_pool = TavilyKeyPool(
                api_keys,
                usage_path or Path(__file__).resolve().parents[1] / "data" / "tavily_key_usage.sqlite3",
                per_key_limit=per_key_limit,
            )
            # Configured official keys must never be sent to a third-party proxy.
            self.api_url = self.OFFICIAL_API_URL
        else:
            self.api_url = api_url or self.DEFAULT_API_URL
        self.max_results = max_results
        self.search_depth = search_depth

    def search(self, query: str) -> List[RetrievedSnippet]:
        """Reserve one search attempt, then retrieve structured snippets.

        Reservations include failed HTTP attempts. Pool errors propagate so a
        depleted search budget cannot silently become an empty retrieval result.
        """
        results: List[RetrievedSnippet] = []
        try:
            payload = {
                "query": query,
                "max_results": self.max_results,
                "search_depth": self.search_depth,
            }
            request_options = {"json": payload, "timeout": 15}
            if self._key_pool is not None:
                payload["auto_parameters"] = False
                response = pooled_search_post(self._key_pool, payload, timeout=15)
            else:
                # Preserve existing single-key/proxy integrations.
                payload["api_key"] = self.api_key
                response = requests.post(self.api_url, **request_options)

            if response.status_code == 200:
                data = response.json()
                for item in data.get("results", []):
                    title = item.get("title", "")
                    url = item.get("url", "")
                    content = item.get("content", "")

                    if title and url and content:
                        results.append(RetrievedSnippet(
                            title=title,
                            url=url,
                            snippet=content,
                            description=content,
                        ))
            else:
                if self.strict_errors:
                    error = TavilySearchRequestError(f"Tavily search returned HTTP {response.status_code}.",
                        status_code=response.status_code, halt_batch=self._key_pool is None and response.status_code in {400,401,402,403,404,405,410,413,422})
                    raise error
                logger.warning(
                    "Tavily search failed for query '%s': HTTP %d",
                    query, response.status_code,
                )
        except TavilyKeyPoolError:
            raise
        except Exception as exc:
            if self.strict_errors:
                if isinstance(exc, TavilySearchRequestError):
                    raise
                raise TavilySearchRequestError(f"Tavily request failed ({type(exc).__name__}); its attempt remains counted.") from None
            # Response bodies and exception messages may contain request secrets.
            status = getattr(exc, "status_code", None)
            logger.warning("Tavily search failed for query '%s': %s%s", query,
                           type(exc).__name__, f" HTTP {status}" if status is not None else "")
        return results

    def search_multiple(self, queries: List[str]) -> List[RetrievedSnippet]:
        """Search multiple queries and deduplicate by URL."""
        seen_urls = set()
        all_results: List[RetrievedSnippet] = []
        for query in queries:
            for snippet in self.search(query):
                if snippet.url not in seen_urls:
                    seen_urls.add(snippet.url)
                    all_results.append(snippet)
        return all_results

    @staticmethod
    def format_snippets(snippets: List[RetrievedSnippet], max_chars: int = 3000) -> str:
        """Format retrieved snippets into a readable context string."""
        return DuckDuckGoRetriever.format_snippets(snippets, max_chars=max_chars)
