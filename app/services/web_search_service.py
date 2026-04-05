from __future__ import annotations

import asyncio
from typing import Protocol

from app.core.exceptions import ConfigurationError, ExternalServiceError, FeatureUnavailableError
from app.core.models import WebSearchResult
from app.utils.text_utils import snippet_preview


class WebSearchProvider(Protocol):
    async def search(self, query: str, limit: int) -> list[WebSearchResult]:
        ...


class DuckDuckGoWebSearchProvider:
    def __init__(self, snippet_max_chars: int) -> None:
        self.snippet_max_chars = snippet_max_chars

    async def search(self, query: str, limit: int) -> list[WebSearchResult]:
        try:
            from duckduckgo_search import DDGS
        except ImportError as exc:  # pragma: no cover - dependency/import failure
            raise FeatureUnavailableError(
                "duckduckgo-search is not installed, so web search is unavailable."
            ) from exc

        def _run_search() -> list[dict]:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=limit))

        try:
            raw_results = await asyncio.to_thread(_run_search)
        except Exception as exc:
            raise ExternalServiceError("Web search request failed.", details=str(exc)) from exc

        normalized: list[WebSearchResult] = []
        for item in raw_results:
            title = str(item.get("title") or "").strip()
            url = str(item.get("href") or item.get("url") or "").strip()
            body = str(item.get("body") or item.get("snippet") or "").strip()
            if not title or not url:
                continue
            normalized.append(
                WebSearchResult(
                    title=title,
                    url=url,
                    snippet=snippet_preview(body, max_chars=self.snippet_max_chars),
                )
            )
        return normalized


class WebSearchService:
    def __init__(
        self,
        *,
        enabled: bool,
        provider_name: str,
        max_results: int,
        snippet_max_chars: int,
    ) -> None:
        self.enabled = enabled
        self.provider_name = provider_name
        self.max_results = max_results
        self.snippet_max_chars = snippet_max_chars

    def _get_provider(self) -> WebSearchProvider:
        if self.provider_name == "duckduckgo":
            return DuckDuckGoWebSearchProvider(snippet_max_chars=self.snippet_max_chars)
        raise ConfigurationError(f"Unsupported web search provider '{self.provider_name}'.")

    async def search(self, query: str, limit: int | None = None) -> list[WebSearchResult]:
        if not self.enabled:
            raise FeatureUnavailableError("Web search is disabled by configuration.")

        provider = self._get_provider()
        bounded_limit = min(limit or self.max_results, self.max_results)
        return await provider.search(query, bounded_limit)
