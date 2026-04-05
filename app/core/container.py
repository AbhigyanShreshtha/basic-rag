from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.config import Settings
    from app.loaders.pdf_loader import PDFLoader
    from app.loaders.role_loader import RoleLoader
    from app.loaders.text_loader import TextLoader
    from app.services.chunking_service import ChunkingService
    from app.services.document_service import DocumentService
    from app.services.embedding_service import EmbeddingService
    from app.services.multimodal_service import MultimodalService
    from app.services.ollama_client import OllamaClient
    from app.services.rag_service import RagService
    from app.services.retrieval_service import RetrievalService
    from app.services.role_service import RoleService
    from app.services.session_service import SessionService
    from app.services.web_search_service import WebSearchService
    from app.storage.metadata_store import SQLiteMetadataStore
    from app.storage.vector_store import ChromaVectorStore


@dataclass
class ServiceContainer:
    settings: "Settings"
    ollama_client: "OllamaClient"
    embedding_service: "EmbeddingService"
    chunking_service: "ChunkingService"
    role_service: "RoleService"
    document_service: "DocumentService"
    retrieval_service: "RetrievalService"
    web_search_service: "WebSearchService"
    multimodal_service: "MultimodalService"
    session_service: "SessionService"
    rag_service: "RagService"
    vector_store: "ChromaVectorStore"
    metadata_store: "SQLiteMetadataStore"
    text_loader: "TextLoader"
    pdf_loader: "PDFLoader"
    role_loader: "RoleLoader"
