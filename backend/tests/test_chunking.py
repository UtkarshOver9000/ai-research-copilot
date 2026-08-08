from copilot.chunking import chunk_text


def test_empty_text_produces_no_chunks():
    assert chunk_text("   ", doc_id="d1", doc_name="Doc") == []


def test_short_text_produces_single_chunk():
    text = "one two three four five"
    chunks = chunk_text(text, doc_id="d1", doc_name="Doc", chunk_size=50, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].chunk_id == "d1::0"


def test_long_text_produces_overlapping_chunks():
    words = [f"word{i}" for i in range(100)]
    text = " ".join(words)
    chunks = chunk_text(text, doc_id="d1", doc_name="Doc", chunk_size=30, overlap=5)

    assert len(chunks) > 1
    # consecutive chunks share the overlap region
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    assert first_words[-5:] == second_words[:5]


def test_chunk_size_must_exceed_overlap():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("some text here", doc_id="d1", doc_name="Doc", chunk_size=10, overlap=10)


def test_chunk_ids_are_sequential_per_document():
    words = [f"w{i}" for i in range(60)]
    chunks = chunk_text(" ".join(words), doc_id="doc_x", doc_name="Doc", chunk_size=20, overlap=5)
    ids = [c.chunk_id for c in chunks]
    assert ids == [f"doc_x::{i}" for i in range(len(chunks))]
