from __future__ import annotations

from typing import Any

import httpx

from app.core.exceptions import ExternalServiceError
from app.core.models import OllamaChatResult


class OllamaClient:
    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        keep_alive: str,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url
        self.keep_alive = keep_alive
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout_seconds,
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def check_health(self) -> bool:
        try:
            response = await self._client.get("tags")
            response.raise_for_status()
            return True
        except httpx.HTTPError:
            return False

    async def embed_texts(self, *, model: str, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        payload = {
            "model": model,
            "input": texts,
            "keep_alive": self.keep_alive,
        }

        try:
            response = await self._client.post("embed", json=payload)
            if response.status_code == 404:
                return await self._legacy_embed_texts(model=model, texts=texts)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ExternalServiceError(
                "Failed to generate embeddings with Ollama.",
                details=str(exc),
            ) from exc

        data = response.json()
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list):
            raise ExternalServiceError("Ollama returned an invalid embedding response.")
        return embeddings

    async def _legacy_embed_texts(self, *, model: str, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            try:
                response = await self._client.post(
                    "embeddings",
                    json={
                        "model": model,
                        "prompt": text,
                        "keep_alive": self.keep_alive,
                    },
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise ExternalServiceError(
                    "Failed to generate embeddings with Ollama.",
                    details=str(exc),
                ) from exc

            data = response.json()
            embedding = data.get("embedding")
            if not isinstance(embedding, list):
                raise ExternalServiceError("Ollama returned an invalid legacy embedding response.")
            embeddings.append(embedding)
        return embeddings

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        think: bool = False,
    ) -> OllamaChatResult:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "think": think,
        }

        try:
            response = await self._client.post("chat", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ExternalServiceError(
                "Failed to generate a chat response with Ollama.",
                details=str(exc),
            ) from exc

        data = response.json()
        message = data.get("message", {})
        content = str(message.get("content") or data.get("response") or "").strip()
        thinking = (
            message.get("thinking")
            or message.get("reasoning")
            or data.get("thinking")
            or data.get("reasoning")
        )
        timings = {
            key: data[key]
            for key in (
                "total_duration",
                "load_duration",
                "prompt_eval_count",
                "prompt_eval_duration",
                "eval_count",
                "eval_duration",
            )
            if key in data
        }

        return OllamaChatResult(
            model=str(data.get("model") or model),
            content=content,
            thinking=str(thinking).strip() if thinking else None,
            timings=timings,
            raw=data,
        )
