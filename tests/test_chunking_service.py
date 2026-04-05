from __future__ import annotations

from app.services.chunking_service import ChunkingService


def test_chunking_creates_multiple_overlapping_chunks() -> None:
    service = ChunkingService(chunk_size=80, chunk_overlap=15)
    text = (
        "Bananas are yellow and grow in clusters near the top of the plant. "
        "They are widely eaten across the world and appear in many desserts.\n\n"
        "Python is a programming language with readable syntax and a large ecosystem."
    )

    chunks = service.chunk_text(text)

    assert len(chunks) >= 2
    assert all(len(chunk) <= 80 for chunk in chunks)
    assert any("many desserts." in chunk for chunk in chunks)
    assert "appear in many" in chunks[1]
    assert "appear in many" in chunks[2]
