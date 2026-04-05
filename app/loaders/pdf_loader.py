from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from app.core.exceptions import BadRequestError


class PDFLoader:
    def load(self, path: Path) -> str:
        try:
            reader = PdfReader(str(path))
        except Exception as exc:  # pragma: no cover - library-specific failure
            raise BadRequestError(f"Failed to open PDF '{path.name}'.", details=str(exc)) from exc

        pages: list[str] = []
        try:
            for page in reader.pages:
                pages.append(page.extract_text() or "")
        except Exception as exc:  # pragma: no cover - library-specific failure
            raise BadRequestError(
                f"Failed to extract text from PDF '{path.name}'.", details=str(exc)
            ) from exc

        return "\n\n".join(pages).strip()
