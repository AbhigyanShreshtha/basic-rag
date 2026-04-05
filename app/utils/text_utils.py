from __future__ import annotations

import re
import textwrap


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def snippet_preview(text: str, max_chars: int = 240) -> str:
    normalized = normalize_whitespace(text)
    if len(normalized) <= max_chars:
        return normalized
    return textwrap.shorten(normalized, width=max_chars, placeholder="...")


def truncate_for_prompt(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3].rstrip()}..."
