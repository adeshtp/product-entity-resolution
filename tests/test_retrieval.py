"""Tests for blocking.py: retrieval, candidate recall, train-only fitting."""

import pandas as pd
import pytest

from entity_resolution.blocking import (
    candidate_recall,
    fit_retriever,
    reduction_ratio,
    remaining_candidate_fraction,
    retrieve,
)


def make_catalog(names, ids=None):
    ids = ids or [str(i) for i in range(len(names))]
    return pd.DataFrame({"id": ids, "name": names, "description": [""] * len(names)})


def test_retrieve_returns_top_k_rows_per_query():
    train_a = make_catalog(["Sony Turntable PSLX350H"], ["1"])
    train_b = make_catalog(["Sony Turntable PSLX350H", "Bose Speaker", "Canon Printer"], ["10", "11", "12"])
    retriever = fit_retriever(train_a, train_b)
    out = retrieve(retriever, train_a, train_b, top_k=2)
    assert len(out) == 2
    assert out["rank"].tolist() == [1, 2]


def test_retrieve_ranks_the_true_match_first():
    train_a = make_catalog(["Sony Turntable PSLX350H"], ["1"])
    train_b = make_catalog(
        ["Canon Inkjet Printer", "Sony Turntable PSLX350H", "Bose Speaker"], ["10", "11", "12"]
    )
    retriever = fit_retriever(train_a, train_b)
    out = retrieve(retriever, train_a, train_b, top_k=3)
    assert out.iloc[0]["idB"] == "11"


def test_retrieve_is_robust_to_spelling_drift():
    # The whole point of char n-grams: these two are the same product written
    # differently, and word-level similarity would miss it.
    train_a = make_catalog(["Sony Turntable PSLX350H Belt Drive System"], ["1"])
    train_b = make_catalog(["Sony PSLX 350H Turntable Belt Drive"], ["10"])
    retriever = fit_retriever(train_a, train_b)
    out = retrieve(retriever, train_a, train_b, top_k=1)
    assert out.iloc[0]["idB"] == "10"
    assert out.iloc[0]["cosine_sim"] > 0.5


def test_retrieve_handles_empty_name():
    train_a = make_catalog([""], ["1"])
    train_b = make_catalog(["Something", "Else"], ["10", "11"])
    retriever = fit_retriever(train_a, train_b)
    out = retrieve(retriever, train_a, train_b, top_k=2)
    assert len(out) == 2
    assert out["cosine_sim"].between(0.0, 1.0).all()


def test_top_k_larger_than_catalog_is_clipped():
    train_a = make_catalog(["A", "B"], ["1", "2"])
    train_b = make_catalog(["X"], ["10"])
    retriever = fit_retriever(train_a, train_b)
    out = retrieve(retriever, train_a, train_b, top_k=20)
    assert len(out) == 2  # one row per A, not 20


def test_fit_retriever_does_not_see_held_out_text():
    """The vectorizer vocabulary must come from train text only."""
    train_a = make_catalog(["alpha bravo"], ["1"])
    train_b = make_catalog(["charlie delta"], ["2"])
    retriever = fit_retriever(train_a, train_b)
    vocabulary = set(retriever.vocabulary_)
    # char_wb vocabulary holds n-grams, not whole words.
    assert "charlie" not in vocabulary
    assert any("char" in gram for gram in vocabulary)
    # An n-gram seen only in held-out text must be unknown to the fitted vectorizer.
    assert not any("zulu" in gram for gram in vocabulary)
    # Held-out text still transforms into the frozen feature space.
    matrix = retriever.transform(["zulu"])
    assert matrix.shape[1] == len(vocabulary)


def test_candidate_recall_counts_a_record_as_hit_if_any_gold_b_is_present():
    candidates = pd.DataFrame(
        {
            "idA": ["a1", "a2"],
            "rank": [1, 1],
            "idB": ["b1", "bX"],
            "cosine_sim": [0.9, 0.5],
        }
    )
    gold = pd.DataFrame({"idAbt": ["a1", "a2"], "idBuy": ["b1", "b2"]})
    result = candidate_recall(candidates, gold, {"a1", "a2"})
    assert result["recall_at_5"] == pytest.approx(0.5)
    assert result["matchable_a_records"] == 2


def test_candidate_recall_ignores_queries_with_no_gold_match():
    candidates = pd.DataFrame({"idA": ["z"], "rank": [1], "idB": ["b1"], "cosine_sim": [0.9]})
    gold = pd.DataFrame({"idAbt": ["a1"], "idBuy": ["b1"]})
    result = candidate_recall(candidates, gold, {"a1"})
    assert result["matchable_a_records"] == 1


def test_reduction_ratio():
    query = make_catalog(["a", "b"], ["1", "2"])
    catalog = make_catalog(["x", "y", "z"], ["10", "11", "12"])
    candidates = pd.DataFrame({"idA": ["1", "2"], "rank": [1, 1], "idB": ["10", "11"], "cosine_sim": [0.1, 0.1]})
    # 2 candidates survive out of a 2 x 3 = 6 grid, so 2/6 of the grid remains
    # and 4/6 was removed.
    assert remaining_candidate_fraction(candidates, query, catalog) == pytest.approx(2 / 6)
    assert reduction_ratio(candidates, query, catalog) == pytest.approx(4 / 6)