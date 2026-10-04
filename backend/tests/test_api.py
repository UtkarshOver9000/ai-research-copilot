from fastapi.testclient import TestClient

from copilot.api.app import app

client = TestClient(app)

DOCS = [
    {
        "name": "rockets.txt",
        "text": (
            "Rockets use controlled combustion of propellant to generate thrust. "
            "Multi-stage rockets discard empty fuel stages to reduce mass during ascent. "
            "Orbital launches require reaching a horizontal velocity sufficient to fall "
            "around the Earth rather than back onto it."
        ),
    },
    {
        "name": "bread.txt",
        "text": (
            "Sourdough bread relies on a fermented starter culture containing wild yeast "
            "and lactobacilli. The long fermentation develops flavor and a chewy crumb "
            "structure distinct from commercially yeasted bread."
        ),
    },
]


def test_health_check():
    res = client.get("/v1/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_query_returns_relevant_citation_first():
    res = client.post("/v1/query", json={"documents": DOCS, "question": "How do rockets reach orbit?", "top_k": 3})
    assert res.status_code == 200
    data = res.json()
    assert data["citations"][0]["doc_name"] == "rockets.txt"
    assert data["generation_available"] is False
    assert data["answer"] is None


def test_query_with_no_documents_returns_no_citations():
    res = client.post("/v1/query", json={"documents": [], "question": "anything"})
    assert res.status_code == 200
    assert res.json()["citations"] == []


def test_query_respects_top_k():
    res = client.post("/v1/query", json={"documents": DOCS, "question": "bread fermentation", "top_k": 1})
    assert res.status_code == 200
    assert len(res.json()["citations"]) <= 1


def test_benchmark_serves_scifact_results():
    res = client.get("/v1/benchmark")
    assert res.status_code == 200
    data = res.json()
    assert data["dataset"] == "BEIR SciFact"
    assert data["retriever"] == "bm25"
    for key in ("ndcg@10", "recall@5", "mrr@10", "precision@1"):
        assert 0.0 < data["test"][key] <= 1.0


def test_top_k_is_validated():
    res = client.post("/v1/query", json={"documents": DOCS, "question": "x", "top_k": 0})
    assert res.status_code == 422


def test_root_serves_html():
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]


def test_query_degrades_gracefully_with_invalid_api_key(monkeypatch):
    # Regression guard: a misconfigured/invalid OPENAI_API_KEY must not take
    # down retrieval, which has no external dependency of its own.
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    res = client.post("/v1/query", json={"documents": DOCS, "question": "How do rockets work?", "top_k": 3})
    assert res.status_code == 200
    data = res.json()
    assert data["generation_available"] is True
    assert data["answer"] is None
    assert len(data["citations"]) > 0


def test_extract_endpoint_handles_text_upload():
    files = {"file": ("notes.txt", b"hello from a plain text file", "text/plain")}
    res = client.post("/v1/extract", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["text"] == "hello from a plain text file"
    assert data["chars"] == len(data["text"])
