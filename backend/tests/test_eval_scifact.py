import json

import pytest

from copilot.eval_scifact import build_chunks, evaluate, load_scifact, query_metrics, rank_documents, write_summary
from copilot.index import make_index


def test_query_metrics_on_known_ranking():
    m = query_metrics(["d3", "d1", "d9", "d2"], {"d1": 1, "d2": 1})
    assert m["recall@1"] == 0.0
    assert m["recall@3"] == 0.5
    assert m["recall@5"] == 1.0
    assert m["precision@3"] == pytest.approx(1 / 3)
    assert m["mrr@10"] == 0.5
    assert m["first_relevant_rank"] == 2
    assert m["success@5"] == 1.0
    # DCG = 1/log2(3) + 1/log2(5); ideal = 1 + 1/log2(3)
    assert m["ndcg@10"] == pytest.approx((1 / 1.5849625 + 1 / 2.3219281) / (1 + 1 / 1.5849625), rel=1e-6)
    assert m["map@100"] == pytest.approx((1 / 2 + 2 / 4) / 2)


def test_no_relevant_document_found():
    m = query_metrics(["a", "b"], {"z": 1})
    assert m["ndcg@10"] == 0.0 and m["mrr@10"] == 0.0 and m["first_relevant_rank"] is None


def _write_beir(tmp_path):
    corpus = [
        {"_id": "1", "title": "Aspirin", "text": "Aspirin reduces the risk of heart attack in adults."},
        {"_id": "2", "title": "Vitamin D", "text": "Vitamin D supplementation and bone fractures in older people."},
        {"_id": "3", "title": "Sleep", "text": "Short sleep duration is associated with obesity in children."},
    ]
    queries = [{"_id": "q1", "text": "aspirin heart attack risk"}, {"_id": "q2", "text": "sleep obesity children"}]
    (tmp_path / "corpus.jsonl").write_text("\n".join(json.dumps(d) for d in corpus))
    (tmp_path / "queries.jsonl").write_text("\n".join(json.dumps(q) for q in queries))
    (tmp_path / "qrels").mkdir()
    (tmp_path / "qrels" / "train.tsv").write_text("query-id\tcorpus-id\tscore\nq1\t1\t1\n")
    (tmp_path / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq2\t3\t1\n")


def test_load_rank_and_evaluate_beir_layout(tmp_path):
    _write_beir(tmp_path)
    corpus, queries, qrels = load_scifact(tmp_path)
    assert set(corpus) == {"1", "2", "3"} and corpus["1"].startswith("Aspirin.")
    assert qrels["train"] == {"q1": {"1": 1}} and qrels["test"] == {"q2": {"3": 1}}

    chunks = build_chunks(corpus)
    doc_ids = [c.doc_id for c in chunks]
    index = make_index("bm25").fit(chunks)
    import numpy as np

    assert rank_documents(index, np.array(doc_ids), "aspirin heart")[0] == "1"
    summary = evaluate(index, np.array(doc_ids), queries, qrels["test"])
    assert summary["queries"] == 1 and summary["recall@1"] == 1.0 and summary["ndcg@10"] == 1.0


def test_shipped_benchmark_summary_is_consistent(tmp_path):
    from copilot.eval_scifact import SUMMARY_PATH

    shipped = json.loads(SUMMARY_PATH.read_text())
    assert shipped["dataset"] == "BEIR SciFact" and shipped["test_claims"] == 300
    assert shipped["retriever"] == max(
        (k for k in shipped["comparison_ndcg@10"] if k != "lsa_previous_default"),
        key=lambda k: shipped["comparison_ndcg@10"][k],
    )
    report = {
        "best_retriever": "bm25",
        "dataset": {"name": "x", "source": "y", "documents": 3, "test_queries": 1},
        "results": {"bm25": {"params": {}, "test": {k: 0.5 for k in shipped["test"]}}},
    }
    out = write_summary(report, tmp_path / "b.json")
    assert json.loads((tmp_path / "b.json").read_text()) == out
