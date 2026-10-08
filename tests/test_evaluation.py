"""Tests for evaluation.py: pair and product metrics, error categories,
coverage, threshold diagnostics, and the frozen test report."""

import json

import numpy as np
import pandas as pd
import pytest

from entity_resolution.blocking import reduction_ratio, remaining_candidate_fraction
from entity_resolution.data import load_splits
from entity_resolution.evaluate_test import FROZEN_MATCH_THRESHOLD, FROZEN_REVIEW_THRESHOLD
from entity_resolution.evaluation import (
    PAIR_RECALL_CEILING,
    abstention_probe,
    classify_outcome,
    end_to_end_metrics,
    gold_coverage_by_decision,
    pair_level_metrics,
    product_level_metrics,
    threshold_diagnostic,
)
from entity_resolution.features import FEATURE_NAMES
from entity_resolution.model import select_match_threshold


def decisions(rows):
    return pd.DataFrame(rows, columns=["idA", "idB", "rank", "decision"])


def test_pair_metrics_on_a_perfect_run():
    gold = pd.DataFrame({"idAbt": ["a1"], "idBuy": ["b1"]})
    decided = decisions([("a1", "b1", 1, "MATCH"), ("a1", "b2", 2, "NO MATCH")])
    metrics = pair_level_metrics(gold, decided, {"a1"})
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["false_merges"] == 0


def test_pair_metrics_count_a_false_merge():
    """A MATCH on a non-gold pair is a false merge, the worst error here."""
    gold = pd.DataFrame({"idAbt": ["a1"], "idBuy": ["b1"]})
    decided = decisions([("a1", "b2", 1, "MATCH")])
    metrics = pair_level_metrics(gold, decided, {"a1"})
    assert metrics["false_merges"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0


def test_pair_metrics_count_a_false_non_match():
    gold = pd.DataFrame({"idAbt": ["a1"], "idBuy": ["b1"]})
    decided = decisions([("a1", "b1", 1, "REVIEW")])
    metrics = pair_level_metrics(gold, decided, {"a1"})
    assert metrics["false_non_matches"] == 1
    assert metrics["recall"] == 0.0


def test_pair_metrics_ignore_records_outside_the_evaluated_set():
    gold = pd.DataFrame({"idAbt": ["a1"], "idBuy": ["b1"]})
    decided = decisions([("a1", "b1", 1, "MATCH"), ("a2", "b9", 1, "MATCH")])
    metrics = pair_level_metrics(gold, decided, {"a1"})
    assert metrics["evaluated_candidate_pairs"] == 1


def test_pair_recall_ceiling_is_the_one_match_per_a_limit():
    # 1097 gold pairs but only 1081 A records, so at most 1081 can be predicted.
    assert PAIR_RECALL_CEILING == pytest.approx(1081 / 1097)
    assert PAIR_RECALL_CEILING < 1.0


def test_product_hits_and_mrr():
    ranked = pd.DataFrame(
        {"idA": ["a1", "a1", "a2"], "rank": [1, 2, 1], "idB": ["bx", "b1", "b2"]}
    )
    gold = pd.DataFrame({"idAbt": ["a1", "a2"], "idBuy": ["b1", "b2"]})
    metrics = product_level_metrics(ranked, gold, {"a1", "a2"})
    assert metrics["hits_at_1"] == pytest.approx(0.5)
    assert metrics["hits_at_5"] == pytest.approx(1.0)
    assert metrics["hits_at_10"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx((1 / 2 + 1 / 1) / 2)


def test_product_metrics_report_records_never_found():
    ranked = pd.DataFrame({"idA": ["a1"], "rank": [1], "idB": ["bx"]})
    gold = pd.DataFrame({"idAbt": ["a1"], "idBuy": ["b1"]})
    metrics = product_level_metrics(ranked, gold, {"a1"})
    assert metrics["hits_at_1"] == 0.0
    assert metrics["never_found"] == 1


def test_product_metrics_skip_a_records_with_no_gold_match():
    ranked = pd.DataFrame({"idA": ["a9"], "rank": [1], "idB": ["bx"]})
    gold = pd.DataFrame({"idAbt": ["a1"], "idBuy": ["b1"]})
    metrics = product_level_metrics(ranked, gold, {"a1", "a9"})
    assert metrics["matchable_a_records"] == 1


def test_end_to_end_recall_multiplies_retrieval_and_matching():
    retrieval = {"recall_at_20": 0.9, "lost_in_blocking_at_20": 5}
    pair = {"recall": 0.5}
    metrics = end_to_end_metrics(None, retrieval, pair)
    assert metrics["end_to_end_recall"] == pytest.approx(0.45)


def test_abstention_probe_detects_a_forced_match():
    """With the true partner gone, a high-scoring rival is a false merge."""
    candidates = pd.DataFrame({"idA": ["a1", "a2"], "idB": ["bx", "by"], "rank": [1, 1]})
    probe = abstention_probe(candidates, np.array([0.99, 0.01]), 0.9)
    assert probe["a_records_probed"] == 2
    assert probe["still_above_match_boundary"] == 1
    assert probe["abstained"] == 1
    assert probe["abstention_rate"] == pytest.approx(0.5)


def test_abstention_probe_handles_no_candidates():
    empty = pd.DataFrame(columns=["idA", "idB"])
    probe = abstention_probe(empty, np.array([]), 0.9)
    assert probe["a_records_probed"] == 0


# ---- leakage guards against the real frozen split ----


def test_split_files_are_the_frozen_ones():
    splits = load_splits()
    counts = {name: len(ids) for name, ids in splits.items()}
    assert counts == {"train": 753, "val": 161, "test": 162}


def test_the_three_splits_do_not_overlap():
    splits = load_splits()
    everything = [g for ids in splits.values() for g in ids]
    assert len(everything) == len(set(everything)) == 1076


def test_ceiling_matches_the_benchmark_structure():
    """1081 A records, 1097 gold pairs: the 16 extra come from 1A->2B groups."""
    assert 1097 - 1081 == 16


# ---- product-level ranking, reduction conventions, gold coverage, diagnostics ----


def catalog(ids):
    return pd.DataFrame({"id": ids})


# ---- item 1: product-level ranking must actually use the chosen ordering ----


def test_product_hits_by_retrieval_rank_reproduce_retrieval_recall():
    """Ranking by retrieval must equal candidate Recall@K, by construction."""
    # gold partner sits at retrieval position 3 for a1, position 1 for a2.
    ranked = pd.DataFrame(
        {
            "idA": ["a1", "a1", "a1", "a2"],
            "idB": ["x1", "x2", "b1", "b2"],
            "rank": [1, 2, 3, 1],
        }
    )
    gold = pd.DataFrame({"idAbt": ["a1", "a2"], "idBuy": ["b1", "b2"]})
    metrics = product_level_metrics(ranked, gold, {"a1", "a2"}, "rank")
    # a1 hits at rank 3 -> inside top-5 and top-10 but not top-1.
    assert metrics["hits_at_1"] == pytest.approx(0.5)
    assert metrics["hits_at_5"] == pytest.approx(1.0)
    assert metrics["hits_at_10"] == pytest.approx(1.0)
    assert metrics["ranked_by"] == "rank"


def test_product_hits_by_model_score_are_computed_in_descending_score_order():
    """The matcher ranking must sort highest score first."""
    ranked = pd.DataFrame(
        {
            "idA": ["a1", "a1", "a1"],
            "idB": ["b1", "x1", "x2"],
            "rank": [3, 1, 2],          # retrieval order disagrees
            "score": [0.4, 0.95, 0.7],  # model order puts b1 LAST
        }
    )
    gold = pd.DataFrame({"idAbt": ["a1"], "idBuy": ["b1"]})
    metrics = product_level_metrics(ranked, gold, {"a1"}, "score")
    # Under model score b1 is worst (0.4), so Hits@1 must be 0.
    assert metrics["hits_at_1"] == 0.0
    assert metrics["hits_at_10"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1 / 3)
    assert metrics["ranked_by"] == "score"


def test_the_two_rankings_can_disagree():
    ranked = pd.DataFrame(
        {
            "idA": ["a1", "a1"],
            "idB": ["b1", "x1"],
            "rank": [2, 1],
            "score": [0.99, 0.10],
        }
    )
    gold = pd.DataFrame({"idAbt": ["a1"], "idBuy": ["b1"]})
    by_rank = product_level_metrics(ranked, gold, {"a1"}, "rank")
    by_score = product_level_metrics(ranked, gold, {"a1"}, "score")
    assert by_rank["hits_at_1"] == 0.0   # retrieval puts the wrong one first
    assert by_score["hits_at_1"] == 1.0  # the matcher gets it right


def test_product_level_rejects_an_unknown_ranking_column():
    ranked = pd.DataFrame({"idA": ["a1"], "idB": ["b1"], "rank": [1]})
    gold = pd.DataFrame({"idAbt": ["a1"], "idBuy": ["b1"]})
    with pytest.raises(KeyError):
        product_level_metrics(ranked, gold, {"a1"}, "not_a_column")


# ---- item 2: both reduction conventions ----


def test_remaining_fraction_and_reduction_ratio_are_complementary():
    query = catalog(["a", "b"])
    pool = catalog(["x", "y", "z"])
    candidates = pd.DataFrame({"idA": ["a", "b"], "idB": ["x", "y"], "rank": [1, 1]})
    remaining = remaining_candidate_fraction(candidates, query, pool)
    reduction = reduction_ratio(candidates, query, pool)
    assert remaining == pytest.approx(2 / 6)
    assert reduction == pytest.approx(1 - 2 / 6)
    assert remaining + reduction == pytest.approx(1.0)


def test_reduction_ratio_is_zero_when_nothing_is_removed():
    query = catalog(["a"])
    pool = catalog(["x"])
    candidates = pd.DataFrame({"idA": ["a"], "idB": ["x"], "rank": [1]})
    assert remaining_candidate_fraction(candidates, query, pool) == pytest.approx(1.0)
    assert reduction_ratio(candidates, query, pool) == pytest.approx(0.0)


# ---- item 5: gold-pair coverage by decision ----


def scored_decisions(rows):
    return pd.DataFrame(rows, columns=["idA", "idB", "rank", "score", "decision"])


def test_gold_coverage_splits_by_decision():
    gold = pd.DataFrame({"idAbt": ["a1", "a2", "a3", "a4"], "idBuy": ["b1", "b2", "b3", "b4"]})
    decided = scored_decisions(
        [
            ("a1", "b1", 1, 0.99, "MATCH"),
            ("a2", "b2", 1, 0.60, "REVIEW"),
            ("a3", "b3", 1, 0.10, "NO MATCH"),
            ("a4", "b4", 1, 0.05, "NO MATCH"),
        ]
    )
    coverage = gold_coverage_by_decision(gold, decided, {"a1", "a2", "a3", "a4"})
    assert coverage["evaluated_gold_pairs"] == 4
    assert coverage["gold_pairs_as_MATCH"] == 1
    assert coverage["gold_pairs_as_REVIEW"] == 1
    assert coverage["gold_pairs_as_NO_MATCH"] == 2
    assert coverage["recall_match_only"] == pytest.approx(0.25)
    assert coverage["recall_match_plus_review"] == pytest.approx(0.5)


def test_false_non_matches_reach_REVIEW_count_separately():
    """41 of 47 gold pairs sat in REVIEW; that must be visible, not hidden."""
    gold = pd.DataFrame({"idAbt": [f"a{i}" for i in range(5)], "idBuy": [f"b{i}" for i in range(5)]})
    decided = scored_decisions(
        [
            ("a0", "b0", 1, 0.99, "MATCH"),
            ("a1", "b1", 1, 0.60, "REVIEW"),
            ("a2", "b2", 1, 0.60, "REVIEW"),
            ("a3", "b3", 1, 0.05, "NO MATCH"),
            ("a4", "b4", 1, 0.05, "NO MATCH"),
        ]
    )
    coverage = gold_coverage_by_decision(gold, decided, {f"a{i}" for i in range(5)})
    assert coverage["false_non_matches_sent_to_REVIEW"] == 2
    assert coverage["false_non_matches_sent_to_NO_MATCH"] == 2
    assert coverage["false_non_matches_sent_to_REVIEW"] + coverage[
        "false_non_matches_sent_to_NO_MATCH"
    ] == 4  # every non-MATCH gold pair


def test_gold_coverage_ignores_non_gold_pairs():
    gold = pd.DataFrame({"idAbt": ["a1"], "idBuy": ["b1"]})
    decided = scored_decisions([("a1", "bX", 1, 0.99, "MATCH")])
    coverage = gold_coverage_by_decision(gold, decided, {"a1"})
    assert coverage["evaluated_gold_pairs"] == 0


# ---- item 3: threshold diagnostic is diagnostic only ----


def test_search_is_not_capped_at_the_precision_target():
    """Regression guard for the corrected search.

    The old implementation swept a 0.05-step grid whose maximum was 0.95, the
    same value as the precision target. That made it impossible for the rule to
    ever satisfy its own target, and it reported 'unattainable' when the target
    was in fact reachable.
    """
    frame = pd.DataFrame(
        {
            "idA": ["a1", "a2", "a3"],
            "idB": ["b1", "bX", "bY"],
            "rank": [1, 1, 1],
            **{name: 0.0 for name in ["name_similarity"]},
        }
    )
    for name in FEATURE_NAMES:
        if name not in frame.columns:
            frame[name] = 0.5
    frame["name_similarity"] = 0.9
    # bY is a false merge at 0.90. Dropping it needs a threshold above 0.96.
    scores = np.array([0.99, 0.96, 0.90])
    gold = {("a1", "b1"), ("a2", "bX")}

    threshold, precision, reached = select_match_threshold(frame, scores, gold, 0.5)
    assert reached is True
    assert threshold > 0.95
    assert precision >= 0.95


def test_selected_threshold_is_an_observed_score():
    """The search must land on an achievable decision point, not an interpolation."""
    frame = pd.DataFrame(
        {
            "idA": ["a1", "a2", "a3"],
            "idB": ["b1", "bX", "bY"],
            "rank": [1, 1, 1],
            **{name: 0.0 for name in ["name_similarity"]},
        }
    )
    for name in FEATURE_NAMES:
        if name not in frame.columns:
            frame[name] = 0.5
    frame["name_similarity"] = 0.9
    scores = np.array([0.9123456789, 0.8765432109, 0.5555555555])
    gold = {("a1", "b1"), ("a2", "bX")}
    threshold, _, _ = select_match_threshold(frame, scores, gold, 0.5)
    assert any(abs(threshold - s) < 1e-12 for s in scores)


def test_reported_threshold_actually_meets_the_target_on_validation():
    """End-to-end: the threshold in metrics.json satisfies its own rule."""
    with open("reports/metrics.json") as handle:
        metrics = json.load(handle)
    thresholds = metrics["thresholds"]
    assert thresholds["target_reached"] is True
    assert thresholds["match_precision_on_validation"] >= thresholds["precision_target"]
    # The selected threshold must not sit exactly on the old defective grid cap.
    assert thresholds["match_threshold"] > 0.95


def test_threshold_diagnostic_reports_metrics_without_selecting():
    frame = pd.DataFrame(
        {
            "idA": ["a1", "a2"],
            "idB": ["b1", "bX"],
            "rank": [1, 1],
            **{name: 0.0 for name in ["x"]},
        }
    )
    scores = np.array([0.99, 0.90])
    gold = {("a1", "b1")}
    rows = threshold_diagnostic(frame, scores, gold, 0.5, [0.5, 0.9, 0.99])
    assert [r["threshold"] for r in rows] == [0.5, 0.9, 0.99]
    for row in rows:
        assert 0.0 <= row["precision"] <= 1.0
        assert 0.0 <= row["recall"] <= 1.0
        assert row["false_merges"] >= 0
    # Stricter threshold -> fewer matches.
    assert rows[0]["match_pairs"] >= rows[-1]["match_pairs"]


# ---- the frozen test report: read it, never regenerate it ----


@pytest.fixture(scope="module")
def report():
    with open("reports/test_metrics.json") as handle:
        return json.load(handle)


# ---- the freeze must hold ----


def test_frozen_thresholds_are_the_documented_values():
    assert FROZEN_MATCH_THRESHOLD == 0.9856394449
    assert abs(FROZEN_REVIEW_THRESHOLD - 0.5182070609) < 1e-9


def test_report_records_the_frozen_block(report):
    frozen = report["frozen"]
    assert frozen["match_threshold"] == FROZEN_MATCH_THRESHOLD
    assert frozen["one_match_per_a"] is True
    assert frozen["candidate_pool_protocol"].startswith("PRIMARY")
    assert "TRAIN" in frozen["retrieval"]
    assert "LogisticRegression" in frozen["model"]


def test_test_split_was_run_exactly_once(report):
    assert report["test_split_evaluation"]["run_count"] == 1


def test_no_match_performance_is_not_claimed(report):
    caveat = report["abstention_probe_caveat"]
    assert "CANNOT be measured" in caveat
    assert "NOT claimed" in caveat


# ---- the report must be internally consistent ----


def test_test_split_sizes_match_the_frozen_split():
    report = json.load(open("reports/test_metrics.json"))
    retrieval = report["retrieval"]
    # 164 test A records x top-20 candidates.
    assert retrieval["matcher_sees_pairs"] == 164 * 20
    assert retrieval["matchable_a_records"] == 164


def test_confusion_matrix_totals_match_the_pair_metrics(report):
    pair = report["pair_level_test"]
    matrix = pair["confusion_matrix"]
    assert matrix[0][1] == pair["false_positives"] == pair["false_merges"]
    assert matrix[1][0] == pair["false_negatives"]
    assert matrix[1][1] == pair["true_positives"]
    assert pair["false_non_matches"] == pair["false_negatives"]


def test_one_match_per_a_holds_on_the_test_split(report):
    queue = report["queue_test"]
    assert queue["a_records_matched"] <= queue["a_records"]
    # Each matched A record contributes at most one MATCH pair.
    assert queue["match_pairs"] <= queue["a_records"]


def test_queue_counts_sum_to_the_candidate_total(report):
    queue = report["queue_test"]
    total = queue["match_pairs"] + queue["review_pairs"] + queue["no_match_pairs"]
    assert total == queue["candidate_pairs"] == 3280
    assert abs(queue["review_queue_rate"] - queue["review_pairs"] / total) < 1e-12


def test_gold_coverage_sums_to_the_evaluated_gold_pairs(report):
    coverage = report["gold_coverage_by_decision_test"]
    assert (
        coverage["gold_pairs_as_MATCH"]
        + coverage["gold_pairs_as_REVIEW"]
        + coverage["gold_pairs_as_NO_MATCH"]
    ) == coverage["evaluated_gold_pairs"] == 164
    assert coverage["recall_match_only"] <= coverage["recall_match_plus_review"]


def test_reduction_conventions_are_complementary(report):
    retrieval = report["retrieval"]
    assert (
        retrieval["remaining_candidate_fraction"] + retrieval["reduction_ratio"]
        == pytest.approx(1.0)
    )


def test_end_to_end_recall_equals_pair_recall_when_no_match_is_lost(report):
    end_to_end = report["end_to_end_test"]
    assert end_to_end["matches_lost_in_blocking"] == 0
    assert end_to_end["end_to_end_recall"] == pytest.approx(end_to_end["pair_recall"])


# ---- bootstrap intervals must be valid ----


def test_bootstrap_intervals_bracket_their_point_estimates(report):
    boot = report["bootstrap_95ci_by_a_product"]
    pair = report["pair_level_test"]
    product = report["product_level_test_model_ranking"]
    for key, point in (
        ("precision", pair["precision"]),
        ("recall", pair["recall"]),
        ("f1", pair["f1"]),
        ("hits_at_1", product["hits_at_1"]),
    ):
        interval = boot[key]
        assert interval["lower_2_5"] <= point <= interval["upper_97_5"], key


def test_bootstrap_metrics_stay_within_zero_and_one(report):
    """Regression guard: an earlier bootstrap divided by false negatives, which
    produced recall and F1 confidence intervals above 1.0."""
    boot = report["bootstrap_95ci_by_a_product"]
    for key in ("precision", "recall", "f1", "hits_at_1"):
        interval = boot[key]
        assert 0.0 <= interval["lower_2_5"] <= interval["upper_97_5"] <= 1.0, key


def test_bootstrap_resamples_by_a_product(report):
    boot = report["bootstrap_95ci_by_a_product"]
    assert "A product" in boot["method"]
    assert boot["units"] == 164
    assert boot["resamples"] == 1000


def test_test_precision_misses_the_target_and_is_reported_not_hidden(report):
    """The registered 0.95 target is NOT met on test. It must be visible."""
    boot = report["bootstrap_95ci_by_a_product"]
    assert report["pair_level_test"]["precision"] < 0.95
    # And the whole interval sits below the target.
    assert boot["precision"]["upper_97_5"] < 0.95


# ---- error analysis uses real rows ----


def test_error_analysis_covers_both_splits_with_real_rows():
    frame = pd.read_csv("reports/error_analysis.csv")
    assert len(frame) > 0
    assert {"split", "error_type", "idA", "idB", "abt_name", "buy_name", "case_tags"} <= set(
        frame.columns
    )
    assert set(frame["split"]) == {"validation", "test"}
    assert frame["abt_name"].str.len().min() > 3
    assert set(frame["error_type"]) <= {
        "false_merge",
        "gold_not_ranked_first",
        "correct_but_sent_to_review",
        "rejected_despite_being_top",
        "gold_never_retrieved",
    }


def test_classify_outcome_distinguishes_retrieval_from_matching_errors():
    gold = {"b1"}
    # Gold was retrieved but the matcher preferred something else.
    assert (
        classify_outcome("bX", "REVIEW", gold, gold_in_candidates=True)
        == "gold_not_ranked_first"
    )
    # Gold was never offered to the matcher at all.
    assert (
        classify_outcome("bX", "REVIEW", gold, gold_in_candidates=False)
        == "gold_never_retrieved"
    )
    # The top candidate WAS gold but needs a human.
    assert (
        classify_outcome("b1", "REVIEW", gold, gold_in_candidates=True)
        == "correct_but_sent_to_review"
    )
    assert (
        classify_outcome("b1", "NO MATCH", gold, gold_in_candidates=True)
        == "rejected_despite_being_top"
    )
    assert classify_outcome("b1", "MATCH", gold, gold_in_candidates=True) == "correct_match"
    assert classify_outcome("bX", "MATCH", gold, gold_in_candidates=True) == "false_merge"


def test_zero_true_retrieval_failures_on_test(report):
    """Candidate Recall@20 is 1.0, so no gold partner was ever missed."""
    frame = pd.read_csv("reports/error_analysis.csv")
    test_rows = frame[frame["split"] == "test"]
    if report["retrieval"]["recall_at_20"] == 1.0:
        assert (test_rows["error_type"] == "gold_never_retrieved").sum() == 0
        assert test_rows["gold_in_candidate_list"].all()


def test_false_merge_counts_match_the_reports(report):
    frame = pd.read_csv("reports/error_analysis.csv")
    test_merges = (frame[frame["split"] == "test"]["error_type"] == "false_merge").sum()
    assert test_merges == report["pair_level_test"]["false_merges"]
