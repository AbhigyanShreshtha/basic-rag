from __future__ import annotations

from app.core.models import RetrievalMode, RetrievedChunk, WebSearchResult
from app.services.embedding_service import EmbeddingService
from app.services.web_search_service import WebSearchService
from app.storage.vector_store import ChromaVectorStore


class RetrievalService:
    def __init__(
        self,
        *,
        embedding_service: EmbeddingService,
        vector_store: ChromaVectorStore,
        web_search_service: WebSearchService,
        default_top_k: int,
        score_threshold: float,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.web_search_service = web_search_service
        self.default_top_k = default_top_k
        self.score_threshold = score_threshold

    async def retrieve_local(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        effective_top_k = top_k or self.default_top_k
        query_embedding = await self.embedding_service.embed_query(query)
        results = self.vector_store.query(query_embedding=query_embedding, top_k=effective_top_k)
        if self.score_threshold > 0:
            results = [chunk for chunk in results if chunk.score >= self.score_threshold]
        return results

    async def retrieve_web(self, query: str, top_k: int | None = None) -> list[WebSearchResult]:
        return await self.web_search_service.search(query, limit=top_k or self.default_top_k)

    async def retrieve(
        self,
        *,
        query: str,
        retrieval_mode: RetrievalMode,
        top_k: int | None = None,
    ) -> tuple[list[RetrievedChunk], list[WebSearchResult]]:
        local_results: list[RetrievedChunk] = []
        web_results: list[WebSearchResult] = []

        if retrieval_mode in {RetrievalMode.local_only, RetrievalMode.hybrid}:
            local_results = await self.retrieve_local(query, top_k=top_k)

        if retrieval_mode in {RetrievalMode.web_only, RetrievalMode.hybrid}:
            web_results = await self.retrieve_web(query, top_k=top_k)

        return local_results, web_results
