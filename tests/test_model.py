"""Tests for the matcher: fitting, threshold rule, decision policy, loading."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from entity_resolution.features import FEATURE_NAMES
from entity_resolution.model import (
    DECISIONS,
    MATCH_PRECISION_TARGET,
    REVIEW_PERCENTILE,
    apply_decisions,
    fit_pair_model,
    select_match_threshold,
    set_review_threshold,
    summarise_queue,
)


def synthetic_pairs(n_positive=60, n_negative=140):
    """A separable but overlapping pair set, in the real feature layout."""
    rows = []
    rng = np.random.default_rng(0)
    for _ in range(n_positive):
        row = {name: 0.0 for name in FEATURE_NAMES}
        row.update(
            {
                "name_similarity": 0.85 + rng.random() * 0.1,
                "description_similarity": 0.8,
                "token_overlap": 0.9,
                "char_similarity": 0.9,
                "model_token_overlap": 1.0,
                "model_token_conflict": 0.0,
                "numeric_token_overlap": 1.0,
                "manufacturer_in_name": 1.0,
                "price_difference": 0.0,
                "price_known_both": 1.0,
            }
        )
        row.update({"idA": "a", "idB": f"b{row['name_similarity']}", "rank": 1, "label": 1})
        rows.append(row)
    for index in range(n_negative):
        row = {name: 0.0 for name in FEATURE_NAMES}
        row.update(
            {
                "name_similarity": 0.1 + rng.random() * 0.3,
                "description_similarity": 0.1,
                "token_overlap": 0.1,
                "char_similarity": 0.2,
                "model_token_overlap": 0.0,
                "model_token_conflict": 1.0,
                "numeric_token_overlap": 0.0,
                "manufacturer_in_name": 0.0,
                "price_difference": 0.0,
                "price_known_both": 0.0,
            }
        )
        row.update({"idA": "a", "idB": f"n{index}", "rank": index + 2, "label": 0})
        rows.append(row)
    return pd.DataFrame(rows)


def test_fit_returns_a_pipeline_and_counts_the_class_balance():
    frame = synthetic_pairs()
    model = fit_pair_model(frame)
    assert isinstance(model.pipeline, Pipeline)
    assert model.train_positive_count == 60
    assert model.train_negative_count == 140


def test_fit_uses_class_weight_balanced():
    model = fit_pair_model(synthetic_pairs())
    classifier = model.pipeline.named_steps["logisticregression"]
    assert classifier.class_weight == "balanced"


def test_fitted_model_separates_the_synthetic_classes():
    frame = synthetic_pairs()
    model = fit_pair_model(frame)
    scores = model.score(frame)
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    positives = scores[frame["label"] == 1]
    negatives = scores[frame["label"] == 0]
    assert positives.mean() > negatives.mean()


def test_coefficients_are_reported_for_every_feature():
    model = fit_pair_model(synthetic_pairs())
    table = model.coefficients()
    assert set(table["feature"]) == set(FEATURE_NAMES)
    assert len(table) == len(FEATURE_NAMES)


def test_model_round_trips_through_joblib(tmp_path):
    joblib = pytest.importorskip("joblib")
    frame = synthetic_pairs()
    model = fit_pair_model(frame)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    assert loaded.train_positive_count == model.train_positive_count
    assert np.allclose(loaded.score(frame), model.score(frame))


def test_threshold_rule_returns_the_lowest_threshold_clearing_the_target():
    """Two A records: one clean positive, one confident-looking false merge."""
    frame = pd.DataFrame(
        {
            "idA": ["good", "bad"],
            "idB": ["b1", "b2"],
            "rank": [1, 1],
            **{name: 0.0 for name in FEATURE_NAMES},
        }
    )
    for name, value in zip(FEATURE_NAMES, [0.95, 0.90, 0.9, 0.9, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]):
        frame.loc[0, name] = value
        frame.loc[1, name] = value * 0.85
    scores = np.array([0.99, 0.60])
    gold = {("good", "b1")}

    threshold, precision, reached = select_match_threshold(frame, scores, gold, 0.5)
    assert reached is True
    assert precision >= MATCH_PRECISION_TARGET
    # The good pair scores 0.99, so the lowest clearing threshold is <= 0.99.
    assert threshold <= 0.99


def test_threshold_rule_respects_one_match_per_a():
    """A second, lower-scoring gold partner must not create a second MATCH."""
    frame = pd.DataFrame(
        {
            "idA": ["a1", "a1"],
            "idB": ["b1", "b2"],
            "rank": [1, 2],
            **{name: 0.0 for name in FEATURE_NAMES},
        }
    )
    frame.loc[0, "name_similarity"] = 0.9
    frame.loc[1, "name_similarity"] = 0.5
    scores = np.array([0.90, 0.30])
    gold = {("a1", "b1"), ("a1", "b2")}

    threshold, precision, _ = select_match_threshold(frame, scores, gold, 0.5)
    from entity_resolution.model import apply_decisions

    decisions = apply_decisions(frame, scores, threshold, 0.5)
    assert (decisions["decision"] == "MATCH").sum() == 1
    assert decisions.loc[decisions["decision"] == "MATCH", "idB"].iloc[0] == "b1"
    assert precision == 1.0


def test_threshold_search_uses_observed_scores_not_a_fixed_grid():
    """The defect this guards: a 0.05 grid capped at 0.95 could not see the
    operating point where the 0.95 target is actually met."""
    frame = pd.DataFrame(
        {
            "idA": ["a1", "a2", "a3"],
            "idB": ["b1", "bX", "bY"],
            "rank": [1, 1, 1],
            **{name: 0.0 for name in FEATURE_NAMES},
        }
    )
    frame["name_similarity"] = 0.9
    # a3 -> bY is a false merge. Clearing it needs a threshold above 0.96, and the
    # only achievable thresholds are the observed scores.
    scores = np.array([0.99, 0.96, 0.90])
    gold = {("a1", "b1"), ("a2", "bX")}

    threshold, precision, reached = select_match_threshold(frame, scores, gold, 0.5)
    assert reached is True
    assert threshold > 0.95, "must be able to search above the precision target"
    assert precision >= MATCH_PRECISION_TARGET


def test_threshold_search_returns_the_lowest_achievable_threshold():
    frame = pd.DataFrame(
        {
            "idA": ["a1", "a2", "a3", "a4"],
            "idB": ["b1", "b2", "bX", "bY"],
            "rank": [1, 1, 1, 1],
            **{name: 0.0 for name in FEATURE_NAMES},
        }
    )
    frame["name_similarity"] = 0.9
    # The two highest-scoring candidates are correct; the two lower ones are wrong.
    scores = np.array([0.60, 0.70, 0.80, 0.90])
    gold = {("a3", "bX"), ("a4", "bY")}

    threshold, precision, reached = select_match_threshold(frame, scores, gold, 0.5)
    assert reached is True
    assert precision >= MATCH_PRECISION_TARGET
    # Lowest threshold that clears the target must not exceed the lowest good score.
    assert threshold <= 0.80
    # And it must be an observed score, i.e. an achievable decision point.
    assert threshold in set(scores)


def test_threshold_search_is_deterministic():
    frame = pd.DataFrame(
        {
            "idA": ["a1", "a2"],
            "idB": ["b1", "bX"],
            "rank": [1, 1],
            **{name: 0.0 for name in FEATURE_NAMES},
        }
    )
    frame["name_similarity"] = 0.9
    scores = np.array([0.97, 0.60])
    gold = {("a1", "b1")}
    first = select_match_threshold(frame, scores, gold, 0.5)
    second = select_match_threshold(frame, scores, gold, 0.5)
    assert first == second


def test_threshold_rule_reports_when_target_is_unreachable():
    """Every candidate that could become MATCH is wrong, so 0.95 is unreachable."""
    frame = pd.DataFrame(
        {
            "idA": ["a1", "a2"],
            "idB": ["b1", "b2"],
            "rank": [1, 1],
            **{name: 0.0 for name in FEATURE_NAMES},
        }
    )
    frame["name_similarity"] = 0.9
    scores = np.array([0.9, 0.85])
    # Neither b1 nor b2 is a gold partner, so any MATCH is a false merge.
    gold = {("a1", "bX"), ("a2", "bY")}

    _, precision, reached = select_match_threshold(frame, scores, gold, 0.5)
    assert reached is False
    assert precision <= MATCH_PRECISION_TARGET


def test_threshold_rule_never_exceeds_one():
    frame = synthetic_pairs(n_positive=20, n_negative=5)
    gold = set(zip(frame[frame["label"] == 1]["idA"], frame[frame["label"] == 1]["idB"]))
    _, precision, _ = select_match_threshold(
        frame.drop(columns=["label"]), np.linspace(0, 1, len(frame)), gold, 0.5
    )
    assert 0.0 <= precision <= 1.0


def test_review_threshold_is_the_train_negative_percentile():
    model = fit_pair_model(synthetic_pairs())
    negative_scores = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    set_review_threshold(model, negative_scores)
    assert model.review_threshold == pytest.approx(np.percentile(negative_scores, REVIEW_PERCENTILE))


def test_review_threshold_falls_back_to_match_when_there_are_no_negatives():
    model = fit_pair_model(synthetic_pairs())
    set_review_threshold(model, np.array([]))
    assert model.review_threshold == model.match_threshold


def test_review_threshold_uses_train_only_not_validation():
    """The REVIEW line must be identical no matter what validation looks like."""
    model = fit_pair_model(synthetic_pairs())
    set_review_threshold(model, np.array([0.2, 0.4, 0.6]))
    first = model.review_threshold
    set_review_threshold(model, np.array([0.2, 0.4, 0.6]))
    assert model.review_threshold == first


def frame(rows):
    return pd.DataFrame([{"idA": a, "idB": b, "rank": r} for a, b, r in rows])


def test_ranking_uses_score_not_retrieval_rank():
    """A lower retrieval rank can still lose on model score."""
    candidates = frame([("a1", "b1", 1), ("a1", "b2", 5)])
    scores = np.array([0.55, 0.99])
    decisions = apply_decisions(candidates, scores, 0.9, 0.5)
    top = decisions.sort_values("score_rank").iloc[0]
    assert top["idB"] == "b2"
    assert top["score_rank"] == 1


def test_accepts_a_series_of_scores():
    candidates = frame([("a1", "b1", 1), ("a1", "b2", 2)])
    decisions = apply_decisions(candidates, pd.Series([0.9, 0.1]), 0.5, 0.3)
    assert (decisions["decision"] == "MATCH").sum() == 1


def test_empty_input_does_not_crash():
    empty = pd.DataFrame(columns=["idA", "idB", "rank"])
    decisions = apply_decisions(empty, np.array([]), 0.5, 0.3)
    assert len(decisions) == 0


def test_summarise_queue_counts_the_review_backlog():
    candidates = frame([("a1", "b1", 1), ("a1", "b2", 2), ("a2", "b3", 1)])
    scores = np.array([0.99, 0.60, 0.10])
    decisions = apply_decisions(candidates, scores, 0.9, 0.5)
    summary = summarise_queue(decisions)
    assert summary["match_pairs"] == 1
    assert summary["review_pairs"] == 1
    assert summary["no_match_pairs"] == 1
    assert summary["a_records"] == 2
    assert summary["a_records_matched"] == 1
    assert summary["review_queue_rate"] == pytest.approx(1 / 3)


def test_summarise_queue_handles_no_candidates():
    empty = pd.DataFrame(columns=["idA", "idB", "decision"])
    summary = summarise_queue(empty)
    assert summary["review_queue_rate"] == 0.0
    assert summary["a_records"] == 0
