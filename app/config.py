from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        protected_namespaces=("settings_",),
    )

    app_name: str = "basic-rag-backend"
    app_env: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    debug_rag: bool = False

    ollama_base_url: str = "http://localhost:11434/api"
    ollama_chat_model: str = "llama3.1:8b"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_keep_alive: str = "5m"
    ollama_timeout_seconds: float = 120.0
    ollama_think: bool = False

    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 4
    score_threshold: float = 0.0
    max_context_chars: int = 12000

    data_dir: Path = DEFAULT_DATA_DIR
    roles_dir: Path | None = None
    chroma_dir: Path | None = None
    uploads_dir: Path | None = None
    metadata_db_path: Path | None = None
    vector_collection_name: str = "rag_documents"

    web_search_enabled: bool = False
    web_search_provider: str = "duckduckgo"
    max_web_results: int = 5
    web_snippet_max_chars: int = 500

    session_max_turns: int = 4
    max_upload_size_mb: int = 20
    max_image_size_mb: int = 10
    enable_role_warnings: bool = True

    model_capabilities_json: str = (
        '{"llava":"text_image","gemma3":"text_image","gemma4":"text_image"}'
    )

    supported_document_extensions: tuple[str, ...] = Field(
        default=(".txt", ".md", ".pdf")
    )
    supported_image_extensions: tuple[str, ...] = Field(
        default=(".png", ".jpg", ".jpeg", ".webp", ".gif")
    )

    @field_validator("ollama_base_url")
    @classmethod
    def normalize_ollama_base_url(cls, value: str) -> str:
        return value.rstrip("/") + "/"

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, value: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 1000)
        if value >= chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return value

    @model_validator(mode="after")
    def populate_paths(self) -> "Settings":
        if self.roles_dir is None:
            self.roles_dir = self.data_dir / "roles"
        if self.chroma_dir is None:
            self.chroma_dir = self.data_dir / "chroma"
        if self.uploads_dir is None:
            self.uploads_dir = self.data_dir / "uploads"
        if self.metadata_db_path is None:
            self.metadata_db_path = self.data_dir / "metadata.db"
        return self

    @property
    def model_capabilities(self) -> dict[str, str]:
        try:
            raw = json.loads(self.model_capabilities_json)
        except json.JSONDecodeError:
            return {}
        return {str(key).strip(): str(value).strip() for key, value in raw.items()}

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def max_image_size_bytes(self) -> int:
        return self.max_image_size_mb * 1024 * 1024

    def ensure_data_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.roles_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_db_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_data_dirs()
    return settings
