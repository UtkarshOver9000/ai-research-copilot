"""
Split raw document text into overlapping word-window chunks for indexing.
"""

from __future__ import annotations

import re

from .schema import Chunk


def _split_words(text: str) -> list[str]:
    return re.split(r"\s+", text.strip())


def chunk_text(
    text: str,
    doc_id: str,
    doc_name: str,
    chunk_size: int = 220,
    overlap: int = 40,
) -> list[Chunk]:
    """Split `text` into overlapping word-window chunks.

    `chunk_size` and `overlap` are in words, not characters, so chunk
    boundaries stay readable regardless of average word length.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    words = [w for w in _split_words(text) if w]
    if not words:
        return []

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    position = 0
    start = 0
    while start < len(words):
        window = words[start : start + chunk_size]
        chunk_text_value = " ".join(window)
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}::{position}",
                doc_id=doc_id,
                doc_name=doc_name,
                position=position,
                text=chunk_text_value,
            )
        )
        position += 1
        start += step

    return chunks
