from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings


KEYWORDS = ["banana", "yellow", "contract", "law", "python", "code", "doctor", "health"]


def keyword_embedding(text: str) -> list[float]:
    lowered = text.lower()
    return [float(lowered.count(keyword)) for keyword in KEYWORDS]


class FakeEmbeddingService:
    async def embed_texts(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        return [keyword_embedding(text) for text in texts]

    async def embed_query(self, query: str, model: str | None = None) -> list[float]:
        return keyword_embedding(query)


class DummyWebSearchService:
    async def search(self, query: str, limit: int | None = None):  # pragma: no cover - trivial
        return []


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    data_dir = tmp_path / "data"
    settings = Settings(
        data_dir=data_dir,
        roles_dir=data_dir / "roles",
        chroma_dir=data_dir / "chroma",
        uploads_dir=data_dir / "uploads",
        metadata_db_path=data_dir / "metadata.db",
        web_search_enabled=False,
        chunk_size=120,
        chunk_overlap=20,
        ollama_base_url="http://testserver/api",
    )
    settings.ensure_data_dirs()
    return settings
