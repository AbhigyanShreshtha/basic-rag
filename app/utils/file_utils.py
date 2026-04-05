from __future__ import annotations

import hashlib
import re
from pathlib import Path

from app.core.exceptions import BadRequestError, UnsupportedMediaTypeError


SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(filename: str) -> str:
    base_name = Path(filename or "upload").name
    cleaned = SAFE_FILENAME_PATTERN.sub("_", base_name).strip("._")
    return cleaned or "upload"


def ensure_allowed_extension(filename: str, allowed_extensions: tuple[str, ...]) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in allowed_extensions:
        raise UnsupportedMediaTypeError(
            f"Unsupported file type '{suffix or 'unknown'}'. Allowed types: {', '.join(allowed_extensions)}."
        )
    return suffix


def guess_content_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    mapping = {
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    return mapping.get(suffix, "application/octet-stream")


def build_storage_path(directory: Path, document_id: str, filename: str) -> Path:
    safe_name = sanitize_filename(filename)
    return directory / f"{document_id}_{safe_name}"


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_file_size(size_bytes: int, max_size_bytes: int, *, kind: str) -> None:
    if size_bytes > max_size_bytes:
        raise BadRequestError(
            f"{kind} exceeds the configured size limit of {max_size_bytes // (1024 * 1024)} MB."
        )
