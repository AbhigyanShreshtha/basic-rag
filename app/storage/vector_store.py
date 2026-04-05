from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.models import RetrievedChunk


class ChromaVectorStore:
    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(
        self,
        *,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(self, *, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        if self.count() == 0:
            return []

        response = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]
        chunks: list[RetrievedChunk] = []

        for index, document_text in enumerate(documents):
            metadata = metadatas[index] or {}
            distance = float(distances[index]) if index < len(distances) else 1.0
            score = max(0.0, 1.0 - distance)
            chunks.append(
                RetrievedChunk(
                    document_id=str(metadata.get("document_id", "")),
                    chunk_id=str(metadata.get("chunk_id", "")),
                    filename=str(metadata.get("filename", "")),
                    text=document_text or "",
                    score=score,
                    metadata=metadata,
                )
            )

        return chunks

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})

    def count(self) -> int:
        return self._collection.count()
