from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional
import requests

logger = logging.getLogger(__name__)


@dataclass
class RetrievedSnippet:
    """A single retrieved search result."""
    title: str
    url: str
    snippet: str
    description: str = ""


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
    ):
        self.max_results = max_results
        self.safe_search = safe_search
        self.region = region
        self._ddgs = None

    def _ensure_ddgs(self):
        if self._ddgs is None:
            try:
                # Prefer the current package name.
                from ddgs import DDGS
                self._ddgs = DDGS()
            except ImportError:
                try:
                    # Retain compatibility with the legacy package name.
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
                backend="api",
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
        if not snippets:
            return "(no search results)"
        lines = []
        total = 0
        for i, s in enumerate(snippets, 1):
            entry = f"[{i}] {s.title}\n    URL: {s.url}\n    {s.snippet}"
            if total + len(entry) > max_chars:
                break
            lines.append(entry)
            total += len(entry)
        return "\n\n".join(lines)


class TavilyRetriever:
    """
    Tavily web retriever for CCA.
    Uses the official Tavily Search API by default.
    """

    DEFAULT_API_URL = "https://api.tavily.com/search"

    def __init__(
        self,
        api_key: str,
        api_url: Optional[str] = None,
        max_results: int = 5,
        search_depth: str = "basic",
    ):
        self.api_key = api_key
        self.api_url = api_url or self.DEFAULT_API_URL
        self.max_results = max_results
        self.search_depth = search_depth

    def search(self, query: str) -> List[RetrievedSnippet]:
        """Search Tavily and return structured snippets."""
        results: List[RetrievedSnippet] = []
        try:
            payload = {
                "api_key": self.api_key,
                "query": query,
                "max_results": self.max_results,
                "search_depth": self.search_depth,
            }

            response = requests.post(
                self.api_url,
                json=payload,
                timeout=15
            )

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
                logger.warning(
                    "Tavily search failed for query '%s': HTTP %d - %s",
                    query, response.status_code, response.text[:200]
                )
        except Exception as e:
            logger.warning("Tavily search failed for query '%s': %s", query, e)
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
        if not snippets:
            return "(no search results)"
        lines = []
        total = 0
        for i, s in enumerate(snippets, 1):
            entry = f"[{i}] {s.title}\n    URL: {s.url}\n    {s.snippet}"
            if total + len(entry) > max_chars:
                break
            lines.append(entry)
            total += len(entry)
        return "\n\n".join(lines)
