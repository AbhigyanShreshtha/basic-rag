from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.models import OllamaChatResult
from app.main import create_app
from tests.conftest import keyword_embedding


def test_query_endpoint_returns_answer_sources_and_debug(settings, monkeypatch) -> None:
    (settings.roles_dir / "coding_assistant.txt").write_text(
        "You are a grounded coding assistant.",
        encoding="utf-8",
    )

    app = create_app(settings)
    with TestClient(app) as client:
        container = client.app.state.container

        async def fake_embed_texts(texts, model=None):
            return [keyword_embedding(text) for text in texts]

        async def fake_embed_query(query, model=None):
            return keyword_embedding(query)

        async def fake_chat(*, model, messages, think=False):
            assert messages[-1]["role"] == "user"
            assert "[Local Context]" in messages[-1]["content"]
            return OllamaChatResult(
                model=model,
                content="Bananas are yellow [L1].",
                thinking="hidden reasoning",
                timings={"total_duration": 10},
            )

        monkeypatch.setattr(container.embedding_service, "embed_texts", fake_embed_texts)
        monkeypatch.setattr(container.embedding_service, "embed_query", fake_embed_query)
        monkeypatch.setattr(container.ollama_client, "chat", fake_chat)

        ingest_response = client.post(
            "/api/v1/documents/ingest",
            files=[
                (
                    "files",
                    ("fruit.txt", b"Bananas are yellow and rich in potassium.", "text/plain"),
                )
            ],
        )
        assert ingest_response.status_code == 200
        assert ingest_response.json()["documents"]

        response = client.post(
            "/api/v1/query",
            files=[
                ("question", (None, "What color are bananas?")),
                ("debug", (None, "true")),
            ],
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["answer"] == "Bananas are yellow [L1]."
        assert payload["sources"][0]["source_id"] == "L1"
        assert payload["sources"][0]["filename"] == "fruit.txt"
        assert payload["debug"]["thinking"] == "hidden reasoning"


def test_query_endpoint_rejects_images_for_text_only_model(settings, monkeypatch) -> None:
    app = create_app(settings)
    with TestClient(app) as client:
        container = client.app.state.container

        async def fake_embed_query(query, model=None):
            return keyword_embedding(query)

        monkeypatch.setattr(container.embedding_service, "embed_query", fake_embed_query)

        response = client.post(
            "/api/v1/query",
            files=[
                ("question", (None, "What is in this image?")),
                ("chat_model", (None, "llama3.1:8b")),
                ("images", ("scan.png", b"fake-image-bytes", "image/png")),
            ],
        )

        assert response.status_code == 400
        assert "text-only" in response.json()["message"]
