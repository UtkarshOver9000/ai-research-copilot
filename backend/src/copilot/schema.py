"""
Core data types for the retrieval pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_name: str
    position: int
    text: str


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass
class QueryResult:
    query: str
    results: list[RetrievedChunk] = field(default_factory=list)
