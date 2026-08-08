from copilot.synthetic_eval import generate_labeled_corpus, run_evaluation


def test_generated_corpus_has_one_answer_chunk_per_query():
    chunks, queries = generate_labeled_corpus(seed=3)
    chunk_ids = {c.chunk_id for c in chunks}
    assert len(queries) > 0
    for q in queries:
        assert q["answer_chunk_id"] in chunk_ids


def test_run_evaluation_reports_well_formed_metrics():
    metrics = run_evaluation(seed=3)
    assert metrics["num_queries"] > 0
    assert 0.0 <= metrics["recall_at_1"] <= 1.0
    assert 0.0 <= metrics["recall_at_10"] <= 1.0
    # recall should be monotonically non-decreasing as k grows
    assert metrics["recall_at_1"] <= metrics["recall_at_3"] <= metrics["recall_at_5"] <= metrics["recall_at_10"]
    assert 0.0 <= metrics["mrr"] <= 1.0


def test_retrieval_beats_a_trivial_baseline():
    # Regression guard: real retrieval should comfortably beat "always wrong"
    # (0.0) and should not be a fluke -- require a strong majority at k=5.
    metrics = run_evaluation(seed=3)
    assert metrics["recall_at_5"] >= 0.8
