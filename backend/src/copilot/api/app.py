"""
FastAPI wrapper around the retrieval pipeline.

Stateless by design: each /v1/query call rebuilds the index from the
documents in that same request rather than depending on server-side
session state, since serverless deployments can't guarantee two requests
land on the same warm instance. Index-fit time is milliseconds for
demo-sized documents, so rebuilding per-query is a fine tradeoff.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from ..chunking import chunk_text
from ..generate import generate_answer, is_generation_available
from ..index import DEFAULT_INDEX, DEFAULT_PARAMS, make_index
from ..ingest import extract_text

BENCHMARK_FILE = Path(__file__).resolve().parent.parent / "benchmark.json"


class DocumentInput(BaseModel):
    name: str
    text: str


class QueryRequest(BaseModel):
    documents: list[DocumentInput]
    question: str
    top_k: int = Field(5, ge=1, le=50)


class CitationResult(BaseModel):
    doc_name: str
    position: int
    text: str
    score: float


class QueryResponse(BaseModel):
    question: str
    citations: list[CitationResult]
    answer: str | None
    generation_available: bool
    timestamp: str


app = FastAPI(
    title="AI Research Copilot API",
    description=(
        "Upload documents, ask a question, get ranked passages with citations. Retrieval is BM25, tuned and "
        "benchmarked on the real BEIR SciFact dataset (see /v1/benchmark), and runs server-side with no "
        "external API calls; generative answers are available when OPENAI_API_KEY is set locally."
    ),
    version="2.0.0",
    contact={"name": "ai-research-copilot", "url": "https://github.com/UtkarshOver9000/ai-research-copilot"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DASHBOARD_DIR = Path(__file__).resolve().parent.parent.parent / "static"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    index_file = DASHBOARD_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file), media_type="text/html")
    return HTMLResponse("<h1>AI Research Copilot API</h1><p>Visit <a href='/docs'>/docs</a></p>")


@app.post("/v1/extract", summary="Extract text from an uploaded file", tags=["Ingestion"])
async def extract(file: UploadFile):
    content = await file.read()
    text = extract_text(file.filename or "upload.txt", content)
    return {"name": file.filename, "text": text, "chars": len(text)}


@app.post("/v1/query", response_model=QueryResponse, summary="Ask a question over documents", tags=["Retrieval"])
async def query(req: QueryRequest):
    all_chunks = []
    for doc_idx, doc in enumerate(req.documents):
        all_chunks.extend(chunk_text(doc.text, doc_id=f"doc_{doc_idx}", doc_name=doc.name))

    if not all_chunks:
        return QueryResponse(
            question=req.question,
            citations=[],
            answer=None,
            generation_available=is_generation_available(),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    index = make_index(DEFAULT_INDEX, **DEFAULT_PARAMS)
    index.fit(all_chunks)
    results = index.query(req.question, top_k=req.top_k)

    citations = [
        CitationResult(
            doc_name=r.chunk.doc_name,
            position=r.chunk.position,
            text=r.chunk.text,
            score=round(r.score, 4),
        )
        for r in results
    ]

    answer = None
    if is_generation_available():
        try:
            answer = generate_answer(req.question, results)
        except Exception:
            # A misconfigured or invalid key shouldn't take down retrieval,
            # which is the part that has no external dependency at all.
            answer = None

    return QueryResponse(
        question=req.question,
        citations=citations,
        answer=answer,
        generation_available=is_generation_available(),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/v1/benchmark", summary="Retrieval quality on the BEIR SciFact benchmark", tags=["Telemetry"])
async def benchmark():
    """Held-out SciFact results for the deployed retriever, produced by `python -m copilot.eval_scifact`."""
    return json.loads(BENCHMARK_FILE.read_text(encoding="utf-8"))


@app.get("/v1/health", include_in_schema=False)
async def health_check():
    return {"status": "ok"}
