from __future__ import annotations

from io import BytesIO

import pytest
from starlette.datastructures import UploadFile

from app.loaders.pdf_loader import PDFLoader
from app.loaders.text_loader import TextLoader
from app.services.chunking_service import ChunkingService
from app.services.document_service import DocumentService
from app.storage.metadata_store import SQLiteMetadataStore
from app.storage.vector_store import ChromaVectorStore
from tests.conftest import FakeEmbeddingService


@pytest.mark.asyncio
async def test_document_ingestion_delete_and_reindex(settings) -> None:
    metadata_store = SQLiteMetadataStore(settings.metadata_db_path)
    vector_store = ChromaVectorStore(settings.chroma_dir, settings.vector_collection_name)
    service = DocumentService(
        settings=settings,
        text_loader=TextLoader(),
        pdf_loader=PDFLoader(),
        chunking_service=ChunkingService(settings.chunk_size, settings.chunk_overlap),
        embedding_service=FakeEmbeddingService(),
        metadata_store=metadata_store,
        vector_store=vector_store,
    )

    upload = UploadFile(
        filename="fruit_notes.txt",
        file=BytesIO(b"Bananas are yellow and nutritious.\n\nBananas are common in desserts."),
    )

    documents, errors = await service.ingest_uploads([upload])
    assert not errors
    assert len(documents) == 1
    assert metadata_store.list_documents()[0].filename == "fruit_notes.txt"
    assert vector_store.count() > 0

    stored_path = documents[0].stored_path
    with open(stored_path, "w", encoding="utf-8") as handle:
        handle.write("Bananas are still yellow after reindexing.")

    updated = await service.reindex_document(documents[0].document_id)
    assert updated.document_id == documents[0].document_id
    assert updated.chunk_count >= 1

    service.delete_document(documents[0].document_id)
    assert metadata_store.list_documents() == []
    assert vector_store.count() == 0

    metadata_store.close()
