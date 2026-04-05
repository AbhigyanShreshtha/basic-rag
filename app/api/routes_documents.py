from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile

from app.api.dependencies import get_container
from app.core.container import ServiceContainer
from app.core.schemas import (
    ActionResponse,
    DocumentIngestResponse,
    DocumentListResponse,
    IngestError,
)


router = APIRouter(tags=["documents"])


@router.post("/documents/ingest", response_model=DocumentIngestResponse)
async def ingest_documents(
    files: list[UploadFile] = File(...),
    container: ServiceContainer = Depends(get_container),
) -> DocumentIngestResponse:
    documents, errors = await container.document_service.ingest_uploads(files)
    return DocumentIngestResponse(
        documents=documents,
        errors=[IngestError(filename=filename, message=message) for filename, message in errors],
    )


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(container: ServiceContainer = Depends(get_container)) -> DocumentListResponse:
    return DocumentListResponse(documents=container.document_service.list_documents())


@router.delete("/documents/{document_id}", response_model=ActionResponse)
def delete_document(
    document_id: str,
    container: ServiceContainer = Depends(get_container),
) -> ActionResponse:
    container.document_service.delete_document(document_id)
    return ActionResponse(status="ok", message=f"Deleted document '{document_id}'.")


@router.post("/documents/{document_id}/reindex", response_model=DocumentIngestResponse)
async def reindex_document(
    document_id: str,
    container: ServiceContainer = Depends(get_container),
) -> DocumentIngestResponse:
    updated = await container.document_service.reindex_document(document_id)
    return DocumentIngestResponse(documents=[updated], errors=[])
