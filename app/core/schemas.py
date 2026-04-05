from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.core.models import DocumentRecord, RetrievalMode, RoleProfile, SourceType


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    details: Any | None = None


class HealthResponse(BaseModel):
    status: str
    app_env: str
    ollama_available: bool
    ollama_base_url: str


class RoleSummary(BaseModel):
    name: str
    description: str | None = None
    tone: str | None = None
    citation_policy: str | None = None


class RolesListResponse(BaseModel):
    roles: list[RoleSummary]


class RolesReloadResponse(BaseModel):
    roles: list[RoleSummary]
    reloaded_count: int


class IngestError(BaseModel):
    filename: str
    message: str


class DocumentIngestResponse(BaseModel):
    documents: list[DocumentRecord]
    errors: list[IngestError] = Field(default_factory=list)


class DocumentListResponse(BaseModel):
    documents: list[DocumentRecord]


class ActionResponse(BaseModel):
    status: str
    message: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    role_name: str | None = None
    retrieval_mode: RetrievalMode = RetrievalMode.local_only
    use_citations: bool = True
    top_k: int | None = Field(default=None, gt=0)
    session_id: str | None = None
    chat_model: str | None = None
    use_thinking: bool | None = None
    debug: bool = False


class SourceResponse(BaseModel):
    source_id: str
    source_type: SourceType
    filename: str | None = None
    url: str | None = None
    chunk_id: str | None = None
    title: str | None = None
    snippet_preview: str
    score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    role_used: str | None = None
    retrieval_mode: RetrievalMode
    session_id: str
    debug: dict[str, Any] | None = None


class QueryJsonRequest(QueryRequest):
    image_base64: list[str] = Field(default_factory=list)
