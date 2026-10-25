"""Independent official DeepSeek Search adapter; no chat or Fetch requests."""
from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
import time
from threading import Lock
from collections import deque
from typing import List
from urllib.parse import urlsplit

import requests

from .retriever import RetrievedSnippet
from .deepseek_client import ProviderConfigurationError
from .storage.persistence import append_jsonl

logger = logging.getLogger(__name__)


class SearchRateLimiter:
    """Process-wide rolling-window limit, shared by all task/key instances."""

    def __init__(self, qps=100, clock=time.monotonic, sleep=time.sleep):
        self.qps, self.clock, self.sleep = qps, clock, sleep
        self.lock, self.starts = Lock(), deque()

    def acquire(self):
        while True:
            with self.lock:
                now = self.clock()
                while self.starts and now - self.starts[0] >= 1.01:
                    self.starts.popleft()
                if len(self.starts) < self.qps:
                    self.starts.append(now)
                    return
                delay = max(0.001, 1.01 - (now - self.starts[0]))
            self.sleep(delay)


class DeepSeekSearchError(RuntimeError):
    """A retrieval failure that must not silently become a no-search result."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.halt_batch = status_code in {400, 401, 402, 403, 404, 405, 410, 413, 422}


class DeepSeekRetriever:
    API_URL = "https://api.deepseek.com/v1/tools/search"
    # QPS controls request starts, not the number of in-flight requests.
    # All keys in this process share the account's Search budget. Separate
    # applications using the same account must share this budget externally.
    _rate = SearchRateLimiter(100)

    def __init__(self, api_key: str | None, *, max_results: int = 5,
                 need_content: bool = False, timeout: float = 20,
                 record_path: Path | None = None, post=None):
        if not api_key or not api_key.strip():
            raise ProviderConfigurationError("DeepSeek search requires DEEPSEEK_SEARCH_API_KEY or DEEPSEEK_API_KEY (official key).")
        if isinstance(max_results, bool) or not isinstance(max_results, int) or not 1 <= max_results <= 50:
            raise ValueError("DeepSeek search max_results must be an integer between 1 and 50.")
        if timeout <= 0:
            raise ValueError("DeepSeek search timeout must be positive.")
        self._api_key = api_key.strip()
        self.max_results = max_results
        self.need_content = need_content
        self.timeout = timeout
        self.record_path = record_path
        self._audit_lock = Lock()
        self._post = post

    @staticmethod
    def _query(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("DeepSeek search query must be non-empty text.")
        text = " ".join(text.split())
        if len(text) > 128:
            prefix = text[:128]
            boundary = prefix.rfind(" ")
            text = prefix[:boundary] if boundary >= 64 else prefix
        return text

    def search(self, query: str) -> List[RetrievedSnippet]:
        for attempt in range(1, 4):
            try:
                return self._search_once(query, attempt)
            except DeepSeekSearchError as exc:
                if not getattr(exc, "retryable_transport", False) or attempt == 3:
                    raise
                time.sleep(0.5 * 2 ** (attempt - 1))
        raise AssertionError("Unreachable search attempt")

    def _search_once(self, query: str, attempt: int) -> List[RetrievedSnippet]:
        sent_query = self._query(query)
        if len(" ".join(query.split())) > 128:
            logger.warning("DeepSeek search query shortened to its 128-character API limit.")
        audit = {
            "timestamp": datetime.now(timezone.utc).isoformat(), "backend": "deepseek",
            "original_query": query, "query": sent_query, "count": self.max_results,
            "need_content": self.need_content, "status_code": None,
            "http_200_with_results": False, "returned_results": 0,
            "attempt": attempt, "process_search_qps_limit": 100,
            "proxy_mode": getattr(self._post, 'proxy_mode', 'environment_default'),
        }
        started = time.monotonic()
        try:
            # Keep TLS verification enabled. Retry only transient SSL transport
            # failures, never certificate verification, authentication or quota.
            self._rate.acquire()
            response = (self._post or requests.post)(
                self.API_URL,
                headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                json={"query": sent_query, "count": self.max_results, "need_content": self.need_content},
                timeout=self.timeout, allow_redirects=False,
            )
            audit["status_code"] = response.status_code
            if response.status_code != 200:
                raise DeepSeekSearchError(f"DeepSeek search returned HTTP {response.status_code}.", response.status_code)
            data = response.json()
            if not isinstance(data, dict) or not isinstance(data.get("results"), list):
                raise DeepSeekSearchError("DeepSeek search returned an invalid results schema.")
            audit["http_200_with_results"] = bool(data["results"])
            audit["returned_results"] = len(data["results"])
            results = []
            seen = set()
            for item in data["results"][:self.max_results]:
                if not isinstance(item, dict):
                    continue
                def text(field):
                    value = item.get(field)
                    return value.strip() if isinstance(value, str) else ""
                url = text("url")
                try:
                    parsed = urlsplit(url)
                    valid_url = parsed.scheme in ("http", "https") and bool(parsed.hostname) and not (parsed.username or parsed.password)
                except ValueError:
                    valid_url = False
                # Provider summaries are suitable for compact model input;
                # full content, if requested, stays in archived metadata.
                body = text("summary") or text("snippet") or text("content")
                if not valid_url or url in seen or not body:
                    continue
                seen.add(url)
                results.append(RetrievedSnippet(
                    title=text("title") or parsed.hostname, url=url, snippet=body,
                    description=text("snippet") or body,
                    metadata={"provider": "deepseek", "query": sent_query,
                              **{key: item.get(key) for key in (
                                  "summary", "snippet", "content", "date", "site_name",
                                  "rerank_score", "authority")}},
                ))
            if data["results"] and not results:
                raise DeepSeekSearchError("DeepSeek search returned results but no usable evidence snippets.")
            audit["usable_results"] = len(results)
            return results
        except DeepSeekSearchError as exc:
            audit["error_type"] = type(exc).__name__
            raise
        except Exception as exc:
            # Never expose bodies, headers or exception strings containing keys.
            audit["error_type"] = type(exc).__name__
            certificate_error = any(term in str(exc).lower() for term in (
                "certificate_verify_failed", "certificate verify failed", "hostname mismatch", "certificate has expired"))
            error = DeepSeekSearchError(f"DeepSeek search failed ({type(exc).__name__}).")
            error.retryable_transport = isinstance(exc, requests.exceptions.SSLError) and not certificate_error
            audit["retryable_transport"] = error.retryable_transport
            raise error from None
        finally:
            audit["elapsed_seconds"] = round(time.monotonic() - started, 6)
            if self.record_path is not None:
                with self._audit_lock:
                    append_jsonl(self.record_path, audit)

    def search_multiple(self, queries: List[str]) -> List[RetrievedSnippet]:
        results = []
        seen_urls = set()
        seen_queries = set()
        for query in queries:
            sent_query = self._query(query)
            if sent_query in seen_queries:
                continue
            seen_queries.add(sent_query)
            for snippet in self.search(query):
                if snippet.url not in seen_urls:
                    seen_urls.add(snippet.url)
                    results.append(snippet)
        return results
