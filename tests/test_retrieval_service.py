from __future__ import annotations

import pytest

from app.services.retrieval_service import RetrievalService
from app.storage.vector_store import ChromaVectorStore
from tests.conftest import DummyWebSearchService, FakeEmbeddingService, keyword_embedding


@pytest.mark.asyncio
async def test_retrieval_service_returns_most_relevant_chunk(settings) -> None:
    vector_store = ChromaVectorStore(settings.chroma_dir, settings.vector_collection_name)
    vector_store.upsert_chunks(
        ids=["doc1:0", "doc2:0"],
        documents=[
            "Bananas are yellow and often used in smoothies.",
            "Contracts usually describe obligations and remedies.",
        ],
        embeddings=[
            keyword_embedding("Bananas are yellow and often used in smoothies."),
            keyword_embedding("Contracts usually describe obligations and remedies."),
        ],
        metadatas=[
            {
                "document_id": "doc1",
                "chunk_id": "doc1:0",
                "filename": "fruit.txt",
                "source_type": "local",
            },
            {
                "document_id": "doc2",
                "chunk_id": "doc2:0",
                "filename": "law.txt",
                "source_type": "local",
            },
        ],
    )

    service = RetrievalService(
        embedding_service=FakeEmbeddingService(),
        vector_store=vector_store,
        web_search_service=DummyWebSearchService(),
        default_top_k=2,
        score_threshold=0.0,
    )

    results = await service.retrieve_local("What fruit is yellow?", top_k=2)

    assert results
    assert results[0].document_id == "doc1"
    assert results[0].filename == "fruit.txt"
