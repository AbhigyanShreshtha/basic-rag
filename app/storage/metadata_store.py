from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import Lock

from app.core.models import DocumentRecord


class SQLiteMetadataStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = Lock()
        self._connection = sqlite3.connect(str(db_path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    ingestion_timestamp TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    checksum TEXT
                )
                """
            )

    def upsert_document(self, record: DocumentRecord) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO documents (
                    document_id,
                    filename,
                    stored_path,
                    content_type,
                    ingestion_timestamp,
                    chunk_count,
                    size_bytes,
                    checksum
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    filename=excluded.filename,
                    stored_path=excluded.stored_path,
                    content_type=excluded.content_type,
                    ingestion_timestamp=excluded.ingestion_timestamp,
                    chunk_count=excluded.chunk_count,
                    size_bytes=excluded.size_bytes,
                    checksum=excluded.checksum
                """,
                (
                    record.document_id,
                    record.filename,
                    record.stored_path,
                    record.content_type,
                    record.ingestion_timestamp.isoformat(),
                    record.chunk_count,
                    record.size_bytes,
                    record.checksum,
                ),
            )

    def list_documents(self) -> list[DocumentRecord]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM documents ORDER BY ingestion_timestamp DESC"
            ).fetchall()
        return [DocumentRecord.model_validate(dict(row)) for row in rows]

    def get_document(self, document_id: str) -> DocumentRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        if row is None:
            return None
        return DocumentRecord.model_validate(dict(row))

    def delete_document(self, document_id: str) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM documents WHERE document_id = ?",
                (document_id,),
            )

    def close(self) -> None:
        self._connection.close()
