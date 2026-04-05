from __future__ import annotations

import httpx
import pytest

from app.services.ollama_client import OllamaClient


@pytest.mark.asyncio
async def test_ollama_client_embed_and_chat_parsing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3]]})
        if request.url.path == "/api/chat":
            return httpx.Response(
                200,
                json={
                    "model": "llama3.1:8b",
                    "message": {
                        "content": "Grounded answer [L1].",
                        "thinking": "internal chain",
                    },
                    "total_duration": 42,
                },
            )
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        return httpx.Response(404)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://testserver/api/",
    ) as http_client:
        client = OllamaClient(
            base_url="http://testserver/api/",
            timeout_seconds=5,
            keep_alive="1m",
            http_client=http_client,
        )

        embeddings = await client.embed_texts(model="nomic-embed-text", texts=["hello"])
        result = await client.chat(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": "Hi"}],
            think=True,
        )

    assert embeddings == [[0.1, 0.2, 0.3]]
    assert result.content == "Grounded answer [L1]."
    assert result.thinking == "internal chain"
    assert result.timings["total_duration"] == 42


@pytest.mark.asyncio
async def test_ollama_client_falls_back_to_legacy_embeddings_endpoint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/embed":
            return httpx.Response(404)
        if request.url.path == "/api/embeddings":
            return httpx.Response(200, json={"embedding": [0.9, 0.8]})
        return httpx.Response(404)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://testserver/api/",
    ) as http_client:
        client = OllamaClient(
            base_url="http://testserver/api/",
            timeout_seconds=5,
            keep_alive="1m",
            http_client=http_client,
        )
        embeddings = await client.embed_texts(model="nomic-embed-text", texts=["first", "second"])

    assert embeddings == [[0.9, 0.8], [0.9, 0.8]]
