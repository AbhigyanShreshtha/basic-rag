from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from fastapi import UploadFile

from app.config import Settings
from app.core.exceptions import AppError, BadRequestError, NotFoundError
from app.core.models import DocumentRecord
from app.loaders.pdf_loader import PDFLoader
from app.loaders.text_loader import TextLoader
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.storage.metadata_store import SQLiteMetadataStore
from app.storage.vector_store import ChromaVectorStore
from app.utils.file_utils import (
    build_storage_path,
    compute_sha256,
    ensure_allowed_extension,
    guess_content_type,
    sanitize_filename,
    validate_file_size,
)
from app.utils.text_utils import normalize_whitespace


class DocumentService:
    def __init__(
        self,
        *,
        settings: Settings,
        text_loader: TextLoader,
        pdf_loader: PDFLoader,
        chunking_service: ChunkingService,
        embedding_service: EmbeddingService,
        metadata_store: SQLiteMetadataStore,
        vector_store: ChromaVectorStore,
    ) -> None:
        self.settings = settings
        self.text_loader = text_loader
        self.pdf_loader = pdf_loader
        self.chunking_service = chunking_service
        self.embedding_service = embedding_service
        self.metadata_store = metadata_store
        self.vector_store = vector_store
        self.logger = logging.getLogger(__name__)

    async def ingest_uploads(
        self, files: Sequence[UploadFile]
    ) -> tuple[list[DocumentRecord], list[tuple[str, str]]]:
        if not files:
            raise BadRequestError("At least one file is required for ingestion.")

        documents: list[DocumentRecord] = []
        errors: list[tuple[str, str]] = []

        for file in files:
            safe_name = sanitize_filename(file.filename or "upload")
            try:
                document = await self._ingest_single(file=file, safe_name=safe_name)
            except AppError as exc:
                message = exc.message
                errors.append((safe_name, message))
                self.logger.warning(
                    "Document ingestion failed",
                    extra={"filename": safe_name, "error": message},
                )
                continue
            documents.append(document)

        return documents, errors

    async def _ingest_single(self, *, file: UploadFile, safe_name: str) -> DocumentRecord:
        ensure_allowed_extension(safe_name, self.settings.supported_document_extensions)
        payload = await file.read()
        validate_file_size(
            len(payload),
            self.settings.max_upload_size_bytes,
            kind=f"Document '{safe_name}'",
        )

        document_id = str(uuid4())
        stored_path = build_storage_path(self.settings.uploads_dir, document_id, safe_name)
        stored_path.write_bytes(payload)

        try:
            text = self._extract_text(stored_path)
            normalized_text = normalize_whitespace(text)
            if not normalized_text:
                raise BadRequestError(f"Document '{safe_name}' did not yield any text content.")

            chunks = self.chunking_service.chunk_text(normalized_text)
            embeddings = await self.embedding_service.embed_texts(chunks)
        except Exception:
            if stored_path.exists():
                stored_path.unlink()
            raise

        timestamp = datetime.now(timezone.utc)
        chunk_ids = [f"{document_id}:{index}" for index in range(len(chunks))]
        metadatas = [
            {
                "document_id": document_id,
                "chunk_id": chunk_ids[index],
                "filename": safe_name,
                "content_type": guess_content_type(safe_name),
                "ingestion_timestamp": timestamp.isoformat(),
                "source_type": "local",
            }
            for index in range(len(chunks))
        ]

        self.vector_store.upsert_chunks(
            ids=chunk_ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        record = DocumentRecord(
            document_id=document_id,
            filename=safe_name,
            stored_path=str(stored_path),
            content_type=guess_content_type(safe_name),
            ingestion_timestamp=timestamp,
            chunk_count=len(chunks),
            size_bytes=len(payload),
            checksum=compute_sha256(payload),
        )
        self.metadata_store.upsert_document(record)
        return record

    def list_documents(self) -> list[DocumentRecord]:
        return self.metadata_store.list_documents()

    def delete_document(self, document_id: str) -> None:
        record = self.metadata_store.get_document(document_id)
        if record is None:
            raise NotFoundError(f"Document '{document_id}' was not found.")

        self.vector_store.delete_document(document_id)
        self.metadata_store.delete_document(document_id)

        path = Path(record.stored_path)
        if path.exists():
            path.unlink()

    async def reindex_document(self, document_id: str) -> DocumentRecord:
        record = self.metadata_store.get_document(document_id)
        if record is None:
            raise NotFoundError(f"Document '{document_id}' was not found.")

        path = Path(record.stored_path)
        if not path.exists():
            raise NotFoundError(
                f"Stored file for document '{document_id}' no longer exists and cannot be reindexed."
            )

        payload = path.read_bytes()
        text = self._extract_text(path)
        normalized_text = normalize_whitespace(text)
        if not normalized_text:
            raise BadRequestError(f"Document '{record.filename}' did not yield any text content.")

        chunks = self.chunking_service.chunk_text(normalized_text)
        embeddings = await self.embedding_service.embed_texts(chunks)

        self.vector_store.delete_document(document_id)

        chunk_ids = [f"{document_id}:{index}" for index in range(len(chunks))]
        metadatas = [
            {
                "document_id": document_id,
                "chunk_id": chunk_ids[index],
                "filename": record.filename,
                "content_type": record.content_type,
                "ingestion_timestamp": record.ingestion_timestamp.isoformat(),
                "source_type": "local",
            }
            for index in range(len(chunks))
        ]
        self.vector_store.upsert_chunks(
            ids=chunk_ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        updated = record.model_copy(
            update={
                "chunk_count": len(chunks),
                "size_bytes": len(payload),
                "checksum": compute_sha256(payload),
            }
        )
        self.metadata_store.upsert_document(updated)
        return updated

    def _extract_text(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            return self.text_loader.load(path)
        if suffix == ".pdf":
            return self.pdf_loader.load(path)
        raise BadRequestError(f"Unsupported document type '{suffix}'.")
