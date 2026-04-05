from __future__ import annotations

import re

from app.utils.text_utils import normalize_whitespace


class ChunkingService:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> list[str]:
        normalized = normalize_whitespace(text)
        if not normalized:
            return []

        chunks: list[str] = []
        start = 0
        while start < len(normalized):
            tentative_end = min(len(normalized), start + self.chunk_size)
            if tentative_end < len(normalized):
                end = self._find_boundary(normalized, start, tentative_end)
            else:
                end = tentative_end

            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= len(normalized):
                break

            next_start = max(0, end - self.chunk_overlap)
            if next_start <= start:
                next_start = end
            start = next_start

        return chunks

    def _find_boundary(self, text: str, start: int, tentative_end: int) -> int:
        if tentative_end >= len(text):
            return len(text)

        window = text[start:tentative_end]
        search_floor = max(0, len(window) // 2)
        boundary_patterns = [r"\n\n", r"\n", r"\. ", r"\! ", r"\? ", r"; ", r", ", r" "]

        for pattern in boundary_patterns:
            matches = list(re.finditer(pattern, window))
            if not matches:
                continue
            last_match = matches[-1]
            if last_match.start() >= search_floor:
                return start + last_match.end()

        return tentative_end
