from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class RetrievalMode(str, Enum):
    local_only = "local_only"
    web_only = "web_only"
    hybrid = "hybrid"


class SourceType(str, Enum):
    local = "local"
    web = "web"


class ModelCapability(str, Enum):
    text_only = "text_only"
    text_image = "text_image"
    embedding_only = "embedding_only"


class RoleProfile(BaseModel):
    name: str
    description: str | None = None
    system_prompt: str
    constraints: list[str] = Field(default_factory=list)
    tone: str | None = None
    citation_policy: str | None = None

    @field_validator("constraints", mode="before")
    @classmethod
    def normalize_constraints(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        raise ValueError("constraints must be a string or list of strings")

    @field_validator("name", "system_prompt")
    @classmethod
    def ensure_non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value cannot be empty")
        return cleaned


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    stored_path: str
    content_type: str
    ingestion_timestamp: datetime
    chunk_count: int = 0
    size_bytes: int = 0
    checksum: str | None = None


class RetrievedChunk(BaseModel):
    document_id: str
    chunk_id: str
    filename: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_type: SourceType = SourceType.local
    source_id: str | None = None


class WebSearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    source_type: SourceType = SourceType.web
    source_id: str | None = None


class MediaAttachment(BaseModel):
    kind: str = "image"
    filename: str
    media_type: str
    base64_data: str


class ChatTurn(BaseModel):
    role: str
    content: str
    timestamp: datetime


class OllamaChatResult(BaseModel):
    model: str
    content: str
    thinking: str | None = None
    timings: dict[str, Any] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)


class AnswerSource(BaseModel):
    source_id: str
    source_type: SourceType
    filename: str | None = None
    url: str | None = None
    chunk_id: str | None = None
    title: str | None = None
    snippet_preview: str
    score: float | None = None


class QueryResult(BaseModel):
    answer: str
    sources: list[AnswerSource]
    role_used: str | None = None
    retrieval_mode: RetrievalMode
    session_id: str
    debug: dict[str, Any] | None = None
