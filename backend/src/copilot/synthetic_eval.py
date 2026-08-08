"""
Synthetic labeled benchmark for the retrieval index.

Generates a corpus of synthetic "documents", each padded with generic
filler paragraphs plus exactly one paragraph containing a unique,
templated fact. A matching natural-language question is generated for
each fact, so the correct source chunk is known ground truth -- this lets
us measure real Recall@k and MRR instead of asserting "retrieval works."
"""

from __future__ import annotations

import argparse
import json
import random

from .chunking import chunk_text
from .index import RetrievalIndex
from .schema import Chunk

PROJECT_WORDS = [
    "Nightingale", "Falcon", "Solstice", "Meridian", "Aurora", "Ember", "Basalt", "Vantage",
    "Halcyon", "Obsidian", "Zephyr", "Cascade", "Lumen", "Kestrel", "Tundra", "Prism",
    "Wraith", "Beacon", "Marlin", "Thicket", "Cobalt", "Ridgeline", "Sable", "Ferrous",
]
PROJECT_PREFIXES = ["Project", "Team", "Pipeline", "Cluster", "Initiative", "Squad"]

# 40 topics built by combining prefixes x words -- many share the same prefix,
# so "Team X" queries have several plausible same-prefix distractors, not just
# unrelated filler paragraphs.
TOPICS = [f"{prefix} {word}" for prefix in PROJECT_PREFIXES for word in PROJECT_WORDS][:40]

METRICS = [
    ("quarterly revenue", "$", "M"),
    ("p99 latency", "", "ms"),
    ("uptime", "", "%"),
    ("headcount", "", " engineers"),
    ("infrastructure budget", "$", "K"),
    ("error rate", "", "%"),
    ("customer churn", "", "%"),
]

PERIODS = ["Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025", "Q1 2026"]

FILLER_SENTENCES = [
    "The team continued its standard weekly sync cadence throughout the period.",
    "No major incidents were reported during the review window.",
    "Documentation was updated to reflect minor process changes.",
    "Several onboarding sessions were held for new contributors.",
    "The roadmap review did not surface any scope changes.",
    "Tooling upgrades were rolled out without user-facing impact.",
    "Cross-team dependencies were tracked in the usual planning board.",
    "A retrospective was held and action items were logged.",
    "Routine maintenance windows were scheduled and completed on time.",
    "Stakeholder updates were circulated on the normal schedule.",
    "The on-call rotation proceeded without escalations.",
    "A handful of minor bug fixes were merged during the period.",
]


def _filler_paragraph(rng: random.Random, sentences: int = 4) -> str:
    return " ".join(rng.choice(FILLER_SENTENCES) for _ in range(sentences))


def generate_labeled_corpus(seed: int = 7) -> tuple[list[Chunk], list[dict]]:
    """Return (all_chunks, queries) where each query dict has
    {"question": str, "answer_chunk_id": str, "topic": str}."""
    rng = random.Random(seed)

    all_chunks: list[Chunk] = []
    queries: list[dict] = []

    for doc_idx, topic in enumerate(TOPICS):
        doc_id = f"doc_{doc_idx}"
        # Cycle metric+period deterministically (rather than sampling freely) so
        # many topics genuinely share the same metric+period phrasing and differ
        # only by topic name and value -- real distractors, not just filler text.
        metric_name, prefix, suffix = METRICS[doc_idx % len(METRICS)]
        period = PERIODS[doc_idx % len(PERIODS)]
        value = rng.randint(10, 999)

        fact_sentence = (
            f"{topic} reported {metric_name} of {prefix}{value}{suffix} in {period}. "
            f"This figure was reviewed by the leads during the {period} planning cycle."
        )

        paragraphs = [_filler_paragraph(rng) for _ in range(5)]
        fact_position = rng.randint(1, len(paragraphs) - 1)
        paragraphs.insert(fact_position, fact_sentence)

        doc_text = "\n\n".join(paragraphs)
        doc_chunks = chunk_text(doc_text, doc_id=doc_id, doc_name=topic, chunk_size=60, overlap=10)
        all_chunks.extend(doc_chunks)

        answer_chunk = next(c for c in doc_chunks if fact_sentence.split(".")[0] in c.text)

        question = f"What was {topic}'s {metric_name} in {period}?"
        queries.append({"question": question, "answer_chunk_id": answer_chunk.chunk_id, "topic": topic})

    return all_chunks, queries


def run_evaluation(seed: int = 7, ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    chunks, queries = generate_labeled_corpus(seed=seed)

    index = RetrievalIndex()
    index.fit(chunks)

    max_k = max(ks)
    ranks: list[int | None] = []  # 1-indexed rank of the correct chunk, or None if outside max_k

    for q in queries:
        results = index.query(q["question"], top_k=max_k)
        result_ids = [r.chunk.chunk_id for r in results]
        if q["answer_chunk_id"] in result_ids:
            ranks.append(result_ids.index(q["answer_chunk_id"]) + 1)
        else:
            ranks.append(None)

    total = len(queries)
    recall_at = {
        f"recall_at_{k}": round(sum(1 for r in ranks if r is not None and r <= k) / total, 4)
        for k in ks
    }
    mrr = round(sum((1.0 / r) if r is not None else 0.0 for r in ranks) / total, 4) if total else 0.0

    return {
        "num_documents": len(TOPICS),
        "num_chunks": len(chunks),
        "num_queries": total,
        **recall_at,
        "mrr": mrr,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the retrieval index on a synthetic labeled corpus")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metrics = run_evaluation(seed=args.seed)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
