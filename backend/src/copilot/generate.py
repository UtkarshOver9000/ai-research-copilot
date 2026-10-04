"""
Optional LLM-based answer synthesis on top of retrieved chunks.

Only active when OPENAI_API_KEY is set. The public live demo runs
retrieval-only (see api/app.py); this is for local use with your own key.
The model can be changed with OPENAI_MODEL.
"""

from __future__ import annotations

import os

from .schema import RetrievedChunk


def is_generation_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def _build_prompt(question: str, results: list[RetrievedChunk]) -> str:
    sources = "\n\n".join(
        f"[{i + 1}] (from {r.chunk.doc_name}, passage {r.chunk.position}): {r.chunk.text}"
        for i, r in enumerate(results)
    )
    return (
        "Answer the question using ONLY the numbered source passages below. "
        "Cite sources inline using their bracket number, e.g. [1]. "
        "If the passages don't contain the answer, say so explicitly.\n\n"
        f"Question: {question}\n\n"
        f"Source passages:\n{sources}\n\n"
        "Answer:"
    )


DEFAULT_MODEL = "gpt-4o-mini"


def generate_answer(question: str, results: list[RetrievedChunk], model: str | None = None) -> str:
    if not is_generation_available():
        raise RuntimeError("OPENAI_API_KEY is not set; generation is unavailable")

    from openai import OpenAI

    client = OpenAI()
    prompt = _build_prompt(question, results)
    response = client.responses.create(model=model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL), input=prompt)
    return response.output_text.strip()
