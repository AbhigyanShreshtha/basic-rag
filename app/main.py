from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes_documents import router as documents_router
from app.api.routes_health import router as health_router
from app.api.routes_query import router as query_router
from app.api.routes_roles import router as roles_router
from app.config import Settings, get_settings
from app.core.container import ServiceContainer
from app.core.exceptions import AppError
from app.core.schemas import ErrorResponse
from app.loaders.pdf_loader import PDFLoader
from app.loaders.role_loader import RoleLoader
from app.loaders.text_loader import TextLoader
from app.logging_config import configure_logging
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


configure_logging()
logger = logging.getLogger(__name__)


def build_container(settings: Settings) -> ServiceContainer:
    settings.ensure_data_dirs()

    text_loader = TextLoader()
    pdf_loader = PDFLoader()
    role_loader = RoleLoader(settings.roles_dir)

    metadata_store = SQLiteMetadataStore(settings.metadata_db_path)
    vector_store = ChromaVectorStore(
        persist_dir=settings.chroma_dir,
        collection_name=settings.vector_collection_name,
    )

    ollama_client = OllamaClient(
        base_url=settings.ollama_base_url,
        timeout_seconds=settings.ollama_timeout_seconds,
        keep_alive=settings.ollama_keep_alive,
    )
    embedding_service = EmbeddingService(ollama_client, settings.ollama_embed_model)
    chunking_service = ChunkingService(settings.chunk_size, settings.chunk_overlap)
    role_service = RoleService(role_loader)
    role_service.load_roles()
    web_search_service = WebSearchService(
        enabled=settings.web_search_enabled,
        provider_name=settings.web_search_provider,
        max_results=settings.max_web_results,
        snippet_max_chars=settings.web_snippet_max_chars,
    )
    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        web_search_service=web_search_service,
        default_top_k=settings.top_k,
        score_threshold=settings.score_threshold,
    )
    multimodal_service = MultimodalService(settings)
    session_service = SessionService(settings.session_max_turns)
    document_service = DocumentService(
        settings=settings,
        text_loader=text_loader,
        pdf_loader=pdf_loader,
        chunking_service=chunking_service,
        embedding_service=embedding_service,
        metadata_store=metadata_store,
        vector_store=vector_store,
    )
    rag_service = RagService(
        settings=settings,
        role_service=role_service,
        retrieval_service=retrieval_service,
        ollama_client=ollama_client,
        session_service=session_service,
        multimodal_service=multimodal_service,
    )

    return ServiceContainer(
        settings=settings,
        ollama_client=ollama_client,
        embedding_service=embedding_service,
        chunking_service=chunking_service,
        role_service=role_service,
        document_service=document_service,
        retrieval_service=retrieval_service,
        web_search_service=web_search_service,
        multimodal_service=multimodal_service,
        session_service=session_service,
        rag_service=rag_service,
        vector_store=vector_store,
        metadata_store=metadata_store,
        text_loader=text_loader,
        pdf_loader=pdf_loader,
        role_loader=role_loader,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        container = build_container(resolved_settings)
        app.state.container = container
        yield
        await container.ollama_client.close()
        container.metadata_store.close()

    app = FastAPI(
        title="Basic Local RAG Backend",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        started_at = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            latency_ms = round((perf_counter() - started_at) * 1000, 2)
            logger.exception(
                "HTTP request failed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "query_params": str(request.query_params),
                    "latency_ms": latency_ms,
                },
            )
            raise
        finally:
            pass

        latency_ms = round((perf_counter() - started_at) * 1000, 2)
        logger.info(
            "HTTP request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
                "status_code": response.status_code,
                "latency_ms": latency_ms,
            },
        )
        return response

    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        logger.warning(
            "Handled application error",
            extra={"error_code": exc.error_code, "details": exc.details},
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error_code=exc.error_code,
                message=exc.message,
                details=exc.details,
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error_code="validation_error",
                message="Request validation failed.",
                details=exc.errors(),
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled server error")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error_code="internal_server_error",
                message="An unexpected server error occurred.",
                details=str(exc),
            ).model_dump(),
        )

    app.include_router(health_router, prefix="/api/v1")
    app.include_router(roles_router, prefix="/api/v1")
    app.include_router(documents_router, prefix="/api/v1")
    app.include_router(query_router, prefix="/api/v1")

    return app


app = create_app()
