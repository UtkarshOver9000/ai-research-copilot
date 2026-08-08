import pytest

from copilot.chunking import chunk_text
from copilot.index import RetrievalIndex


def _sample_chunks():
    docs = {
        "cats": "Cats are small domesticated carnivorous mammals. They are popular pets known for independence.",
        "rockets": "Rockets use controlled combustion to generate thrust for spaceflight and orbital launches.",
        "bread": "Sourdough bread is made by fermenting dough using naturally occurring lactobacilli and yeast.",
    }
    chunks = []
    for doc_id, text in docs.items():
        chunks.extend(chunk_text(text, doc_id=doc_id, doc_name=doc_id, chunk_size=50, overlap=5))
    return chunks


def test_fit_requires_at_least_one_chunk():
    index = RetrievalIndex()
    with pytest.raises(ValueError):
        index.fit([])


def test_query_before_fit_raises():
    index = RetrievalIndex()
    with pytest.raises(RuntimeError):
        index.query("anything")


def test_query_ranks_the_relevant_document_first():
    index = RetrievalIndex()
    index.fit(_sample_chunks())

    results = index.query("How do rockets generate thrust for launches?", top_k=3)
    assert results[0].chunk.doc_id == "rockets"


def test_query_returns_at_most_top_k_results():
    index = RetrievalIndex()
    index.fit(_sample_chunks())
    results = index.query("bread", top_k=2)
    assert len(results) <= 2


def test_scores_are_sorted_descending():
    index = RetrievalIndex()
    index.fit(_sample_chunks())
    results = index.query("cats and pets", top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_index_size_matches_fitted_chunks():
    chunks = _sample_chunks()
    index = RetrievalIndex()
    index.fit(chunks)
    assert index.size == len(chunks)
