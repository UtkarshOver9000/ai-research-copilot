"""
Retrieval benchmark on BEIR SciFact: 5,183 real scientific abstracts and
1,109 expert-written claims with human relevance judgements.

    python -m copilot.eval_scifact --data-dir ../data/scifact

Protocol
--------
1. Every abstract goes through the same chunker the API uses; a document's
   score is its best chunk's score.
2. Hyperparameters of each retriever (BM25 k1/b, LSA components, TF-IDF
   weighting) are tuned on the 809 *train* claims only.
3. The tuned retrievers are scored once on the 300 held-out *test* claims
   (the standard BEIR split) and, for completeness, on all 1,109 claims.

Retrieval has no gradient training, so there are no epochs or loss curves;
the tuning curves on the train split play that role and are saved as figures.
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import sklearn

from .chunking import chunk_text
from .index import make_index

KS = (1, 3, 5, 10, 100)


def load_scifact(data_dir: Path) -> tuple[dict, dict, dict]:
    data_dir = Path(data_dir)
    corpus = {}
    with open(data_dir / "corpus.jsonl", encoding="utf-8") as fh:
        for line in fh:
            doc = json.loads(line)
            corpus[doc["_id"]] = f"{doc.get('title', '')}. {doc.get('text', '')}".strip()
    queries = {}
    with open(data_dir / "queries.jsonl", encoding="utf-8") as fh:
        for line in fh:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    qrels = {}
    for split in ("train", "test"):
        rel = defaultdict(dict)
        with open(data_dir / "qrels" / f"{split}.tsv", encoding="utf-8") as fh:
            next(fh)  # header
            for line in fh:
                qid, did, score = line.rstrip("\n").split("\t")
                rel[qid][did] = int(score)
        qrels[split] = dict(rel)
    return corpus, queries, qrels


def build_chunks(corpus: dict) -> list:
    chunks = []
    for doc_id, text in corpus.items():
        chunks.extend(chunk_text(text, doc_id=doc_id, doc_name=doc_id))
    return chunks


def rank_documents(index, chunk_doc_ids: np.ndarray, query: str, depth: int = 100) -> list[str]:
    """Doc ids ranked by their best chunk score."""
    scores = index.scores(query)
    order = np.argsort(-scores, kind="stable")
    seen, ranked = set(), []
    for i in order:
        doc = chunk_doc_ids[i]
        if doc not in seen:
            seen.add(doc)
            ranked.append(doc)
            if len(ranked) == depth:
                break
    return ranked


def _dcg(gains) -> float:
    return float(sum(g / np.log2(i + 2) for i, g in enumerate(gains)))


def query_metrics(ranked: list[str], relevant: dict) -> dict:
    rel = set(relevant)
    hits = [d in rel for d in ranked]
    out = {}
    for k in KS:
        found = sum(hits[:k])
        out[f"recall@{k}"] = found / len(rel)
        out[f"precision@{k}"] = found / k
    ideal = sorted(relevant.values(), reverse=True)[:10]
    out["ndcg@10"] = _dcg([relevant.get(d, 0) for d in ranked[:10]]) / _dcg(ideal) if ideal else 0.0
    first = next((i for i, h in enumerate(hits) if h), None)
    out["mrr@10"] = 1.0 / (first + 1) if first is not None and first < 10 else 0.0
    precisions = [sum(hits[: i + 1]) / (i + 1) for i, h in enumerate(hits[:100]) if h]
    out["map@100"] = sum(precisions) / len(rel)
    out["success@5"] = float(any(hits[:5]))
    out["first_relevant_rank"] = (first + 1) if first is not None else None
    return out


def evaluate(index, chunk_doc_ids, queries: dict, qrels: dict) -> dict:
    per_query, latencies = [], []
    for qid, relevant in qrels.items():
        t0 = time.perf_counter()
        ranked = rank_documents(index, chunk_doc_ids, queries[qid])
        latencies.append((time.perf_counter() - t0) * 1000)
        per_query.append(query_metrics(ranked, relevant))
    keys = [k for k in per_query[0] if k != "first_relevant_rank"]
    summary = {k: round(float(np.mean([m[k] for m in per_query])), 4) for k in keys}
    ranks = [m["first_relevant_rank"] for m in per_query]
    found = [r for r in ranks if r is not None]
    summary["queries"] = len(per_query)
    summary["hit@1 (top result relevant)"] = summary["precision@1"]
    summary["median_rank_of_first_relevant"] = float(np.median(found)) if found else None
    summary["queries_with_no_relevant_in_top100"] = sum(r is None for r in ranks)
    summary["latency_ms_p50"] = round(float(np.percentile(latencies, 50)), 2)
    summary["latency_ms_p95"] = round(float(np.percentile(latencies, 95)), 2)
    return summary


SEARCH_SPACES = {
    "bm25": [{"k1": k1, "b": b} for k1, b in itertools.product((0.6, 0.9, 1.2, 1.5, 2.0), (0.3, 0.5, 0.75, 0.9))],
    "tfidf": [{"sublinear_tf": True}, {"sublinear_tf": False}],
    "lsa": [{"n_components": n} for n in (50, 100, 150, 200, 300, 400, 600, 800)],
}


def tune(name: str, chunks, chunk_doc_ids, queries, train_qrels) -> tuple[dict, list]:
    curve = []
    for params in SEARCH_SPACES[name]:
        index = make_index(name, **params).fit(chunks)
        score = evaluate(index, chunk_doc_ids, queries, train_qrels)["ndcg@10"]
        curve.append({"params": params, "train_ndcg@10": score})
    best = max(curve, key=lambda c: c["train_ndcg@10"])["params"]
    return best, curve


def save_figures(tuning: dict, results: dict, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    lsa = tuning["lsa"]["curve"]
    axes[0].plot([c["params"]["n_components"] for c in lsa], [c["train_ndcg@10"] for c in lsa], marker="o")
    axes[0].set(title="LSA: train nDCG@10 vs SVD components", xlabel="components", ylabel="nDCG@10")

    bm = tuning["bm25"]["curve"]
    k1s = sorted({c["params"]["k1"] for c in bm})
    bs = sorted({c["params"]["b"] for c in bm})
    grid = np.array(
        [[next(c["train_ndcg@10"] for c in bm if c["params"] == {"k1": k1, "b": b}) for b in bs] for k1 in k1s]
    )
    im = axes[1].imshow(grid, cmap="viridis", aspect="auto")
    axes[1].set_xticks(range(len(bs)), bs)
    axes[1].set_yticks(range(len(k1s)), k1s)
    axes[1].set(title="BM25: train nDCG@10 over k1 x b", xlabel="b", ylabel="k1")
    fig.colorbar(im, ax=axes[1])

    names = list(results)
    ks = [1, 3, 5, 10, 100]
    for name in names:
        axes[2].plot(ks, [results[name]["test"][f"recall@{k}"] for k in ks], marker="o", label=name)
    axes[2].set_xscale("log")
    axes[2].set(title="Held-out test: recall@k", xlabel="k", ylabel="recall")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "scifact_tuning_and_recall.png", dpi=130)
    plt.close(fig)


def run(data_dir: Path, out: Path) -> dict:
    t_start = time.time()
    corpus, queries, qrels = load_scifact(data_dir)
    chunks = build_chunks(corpus)
    chunk_doc_ids = np.array([c.doc_id for c in chunks])
    all_qrels = {**qrels["train"], **qrels["test"]}

    tuning, results = {}, {}
    for name in ("bm25", "tfidf", "lsa"):
        best, curve = tune(name, chunks, chunk_doc_ids, queries, qrels["train"])
        t0 = time.perf_counter()
        index = make_index(name, **best).fit(chunks)
        build_s = time.perf_counter() - t0
        tuning[name] = {"best_params": best, "curve": curve}
        results[name] = {
            "params": best,
            "index_build_seconds": round(build_s, 2),
            "test": evaluate(index, chunk_doc_ids, queries, qrels["test"]),
            "all_labelled_queries": evaluate(index, chunk_doc_ids, queries, all_qrels),
        }

    # The API's previous configuration, for comparison.
    previous = make_index("lsa", n_components=150).fit(chunks)
    results["lsa_previous_default"] = {
        "params": {"n_components": 150},
        "test": evaluate(previous, chunk_doc_ids, queries, qrels["test"]),
    }

    best_name = max(("bm25", "tfidf", "lsa"), key=lambda n: results[n]["test"]["ndcg@10"])
    best_test = results[best_name]["test"]
    report = {
        "dataset": {
            "name": "BEIR SciFact",
            "source": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
            "paper": "Wadden et al., 'Fact or Fiction: Verifying Scientific Claims', EMNLP 2020; BEIR, NeurIPS 2021",
            "documents": len(corpus),
            "chunks": len(chunks),
            "train_queries": len(qrels["train"]),
            "test_queries": len(qrels["test"]),
        },
        "protocol": "hyperparameters tuned on train claims only; metrics below are on held-out test claims",
        "best_retriever": best_name,
        "results": results,
        "tuning": tuning,
        "business": {
            "claims_with_a_relevant_abstract_in_top_5": best_test["success@5"],
            "median_results_read_before_first_relevant": best_test["median_rank_of_first_relevant"],
            "p95_query_latency_ms_over_5k_abstracts": best_test["latency_ms_p95"],
            "index_build_seconds_5k_abstracts": results[best_name]["index_build_seconds"],
            "external_api_cost_per_query_usd": 0.0,
        },
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
        "runtime_seconds": None,
    }
    save_figures(tuning, {k: v for k, v in results.items() if k != "lsa_previous_default"}, out.parent / "figures")
    report["runtime_seconds"] = round(time.time() - t_start, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return report


SUMMARY_PATH = Path(__file__).resolve().parent / "benchmark.json"
SUMMARY_KEYS = (
    "ndcg@10",
    "recall@1",
    "recall@5",
    "recall@10",
    "recall@100",
    "precision@1",
    "precision@5",
    "mrr@10",
    "map@100",
    "success@5",
    "latency_ms_p50",
    "latency_ms_p95",
)


def write_summary(report: dict, path: Path = SUMMARY_PATH) -> dict:
    """Compact copy of the headline numbers that the API serves at /v1/benchmark."""
    best = report["best_retriever"]
    summary = {
        "dataset": report["dataset"]["name"],
        "source": report["dataset"]["source"],
        "documents": report["dataset"]["documents"],
        "test_claims": report["dataset"]["test_queries"],
        "retriever": best,
        "params": report["results"][best]["params"],
        "test": {k: report["results"][best]["test"][k] for k in SUMMARY_KEYS},
        "comparison_ndcg@10": {name: r["test"]["ndcg@10"] for name, r in report["results"].items()},
    }
    path.write_text(json.dumps(summary, indent=2) + chr(10))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="BEIR SciFact retrieval benchmark")
    parser.add_argument("--data-dir", type=Path, default=Path("../data/scifact"))
    parser.add_argument("--out", type=Path, default=Path("../reports/scifact_metrics.json"))
    args = parser.parse_args()
    report = run(args.data_dir, args.out)
    write_summary(report)
    print(json.dumps({k: v["test"] for k, v in report["results"].items()}, indent=2))
    print(json.dumps(report["business"], indent=2))


if __name__ == "__main__":
    main()
