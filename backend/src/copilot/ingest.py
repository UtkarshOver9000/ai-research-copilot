"""
Extract plain text from uploaded document bytes (PDF or plain text).
"""

from __future__ import annotations

import io


def extract_text(filename: str, content: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)

    return content.decode("utf-8", errors="replace")
