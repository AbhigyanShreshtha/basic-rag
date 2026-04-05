from __future__ import annotations

from app.services.ollama_client import OllamaClient


class EmbeddingService:
    def __init__(self, ollama_client: OllamaClient, default_model: str) -> None:
        self.ollama_client = ollama_client
        self.default_model = default_model

    async def embed_texts(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        return await self.ollama_client.embed_texts(model=model or self.default_model, texts=texts)

    async def embed_query(self, query: str, model: str | None = None) -> list[float]:
        embeddings = await self.embed_texts([query], model=model)
        return embeddings[0]
