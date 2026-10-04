import pytest

from copilot.chunking import chunk_text
from copilot.index import DEFAULT_INDEX, DEFAULT_PARAMS, INDEXES, BM25Index, make_index

DOCS = {
    "cats": "Cats are small domesticated carnivorous mammals. They are popular pets known for independence.",
    "rockets": "Rockets use controlled combustion to generate thrust for spaceflight and orbital launches.",
    "bread": "Sourdough bread is made by fermenting dough using naturally occurring lactobacilli and yeast.",
}


def _chunks():
    out = []
    for doc_id, text in DOCS.items():
        out.extend(chunk_text(text, doc_id=doc_id, doc_name=doc_id, chunk_size=50, overlap=5))
    return out


@pytest.mark.parametrize("name", list(INDEXES))
def test_fit_requires_chunks(name):
    with pytest.raises(ValueError):
        make_index(name).fit([])


@pytest.mark.parametrize("name", list(INDEXES))
def test_query_before_fit_raises(name):
    with pytest.raises(RuntimeError):
        make_index(name).query("anything")


@pytest.mark.parametrize("name", list(INDEXES))
def test_relevant_document_ranks_first(name):
    index = make_index(name).fit(_chunks())
    results = index.query("How do rockets generate thrust for launches?", top_k=3)
    assert results[0].chunk.doc_id == "rockets"
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("name", list(INDEXES))
def test_top_k_is_respected(name):
    index = make_index(name).fit(_chunks())
    assert len(index.query("bread", top_k=2)) == 2
    assert len(index.query("bread", top_k=50)) == index.size == 3


def test_default_index_uses_tuned_bm25_params():
    index = make_index()
    assert DEFAULT_INDEX == "bm25" and isinstance(index, BM25Index)
    assert (index.k1, index.b) == (DEFAULT_PARAMS["k1"], DEFAULT_PARAMS["b"])


def test_bm25_rewards_term_matches_and_ignores_missing_terms():
    index = BM25Index().fit(_chunks())
    scores = index.scores("sourdough yeast")
    assert scores.argmax() == 2  # bread chunk
    assert index.scores("zzzz unknownterm").max() == 0.0


def test_unknown_index_name():
    with pytest.raises(ValueError):
        make_index("nope")
