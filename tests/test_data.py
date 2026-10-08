"""Tests for data.py and the end-to-end leakage guards on the frozen split."""

import json
from pathlib import Path

import pandas as pd
import pytest
from joblib import load

from entity_resolution.blocking import candidate_recall, fit_retriever, retrieve
from entity_resolution.data import (
    SPLIT_DIR,
    entity_groups,
    gold_in_split,
    load_all,
    load_splits,
    make_splits,
    records_in_split,
    save_splits,
    split_records,
)
from entity_resolution.features import build_feature_frame, feature_matrix
from entity_resolution.model import apply_decisions, fit_pair_model
from entity_resolution.pairs import mine_training_pairs
from entity_resolution.features import fit_similarity


@pytest.fixture(scope="module")
def loaded():
    return load_all()


def test_row_counts_match_official_page(loaded):
    abt, buy, gold = loaded
    assert len(abt) == 1081
    assert len(buy) == 1092
    assert len(gold) == 1097


def test_expected_columns(loaded):
    abt, buy, gold = loaded
    assert list(abt.columns) == ["id", "name", "description", "price"]
    assert list(buy.columns) == ["id", "name", "description", "manufacturer", "price"]
    assert list(gold.columns) == ["idAbt", "idBuy"]


def test_ids_are_unique(loaded):
    abt, buy, _ = loaded
    assert abt["id"].is_unique
    assert buy["id"].is_unique


def test_names_are_never_empty(loaded):
    abt, buy, _ = loaded
    assert (abt["name"].str.strip() != "").all()
    assert (buy["name"].str.strip() != "").all()


def test_every_record_has_a_gold_match(loaded):
    abt, buy, gold = loaded
    assert set(abt["id"]) == set(gold["idAbt"])
    assert set(buy["id"]) == set(gold["idBuy"])


def test_entity_group_count(loaded):
    _, _, gold = loaded
    assert len(set(entity_groups(gold).values())) == 1076


def test_split_files_cover_every_group_without_overlap(loaded):
    _, _, gold = loaded
    splits = load_splits()
    all_groups = [g for ids in splits.values() for g in ids]
    assert len(all_groups) == len(set(all_groups)) == 1076
    assert set(all_groups) == set(entity_groups(gold).values())


def test_no_gold_pair_crosses_a_split(loaded):
    _, _, gold = loaded
    groups = entity_groups(gold)
    splits = load_splits()
    owner = {g: name for name, ids in splits.items() for g in ids}
    for id_a, id_b in zip(gold["idAbt"], gold["idBuy"]):
        group = groups[("A", id_a)]
        assert group == groups[("B", id_b)]
        assert owner[group] == owner[groups[("B", id_b)]]


def test_split_is_reproducible(loaded):
    _, _, gold = loaded
    assert make_splits(gold, (0.7, 0.15, 0.15), 42) == load_splits()


def test_ratios_must_sum_to_one(loaded):
    _, _, gold = loaded
    with pytest.raises(ValueError):
        make_splits(gold, (0.7, 0.15, 0.2), 42)


def test_save_and_reload_roundtrip(tmp_path, loaded):
    _, _, gold = loaded
    splits = make_splits(gold, (0.7, 0.15, 0.15), 42)
    save_splits(splits, tmp_path)
    reloaded = {
        name: pd.read_csv(tmp_path / f"{name}_groups.csv")["entity_group"].tolist()
        for name in ("train", "val", "test")
    }
    assert reloaded == splits


def test_loader_reports_a_clear_error_when_raw_data_is_missing(tmp_path):
    from entity_resolution import data as data_module

    original = data_module.ABT_FILE
    try:
        data_module.ABT_FILE = tmp_path / "Abt.csv"
        with pytest.raises(FileNotFoundError):
            data_module.load_abt()
    finally:
        data_module.ABT_FILE = original


def test_split_files_exist():
    for name in ("train", "val", "test"):
        assert (Path(SPLIT_DIR) / f"{name}_groups.csv").exists()


# ---- end-to-end: the split and pipeline stay leakage-free ----


@pytest.fixture(scope="module")
def everything():
    abt, buy, gold = split_records()
    splits = load_splits()
    train_a = records_in_split(abt, splits, "train")
    train_b = records_in_split(buy, splits, "train")
    val_a = records_in_split(abt, splits, "val")
    val_gold = gold_in_split(gold, abt, splits, "val")

    retriever = fit_retriever(train_a, train_b)
    train_candidates = retrieve(retriever, train_a, train_b, top_k=20)
    val_candidates = retrieve(retriever, val_a, buy, top_k=20)
    similarity_model = fit_similarity(train_a, train_b)
    train_pairs = mine_training_pairs(
        train_candidates, train_a, train_b, gold, similarity_model
    )
    return {
        "abt": abt,
        "buy": buy,
        "splits": splits,
        "train_a": train_a,
        "train_b": train_b,
        "val_a": val_a,
        "val_gold": val_gold,
        "retriever": retriever,
        "val_candidates": val_candidates,
        "similarity_model": similarity_model,
        "train_pairs": train_pairs,
    }


def test_split_sizes_are_frozen(everything):
    assert len(everything["train_a"]) == 756
    assert len(everything["train_b"]) == 764
    assert len(everything["val_a"]) == 161
    assert len(everything["val_gold"]) == 166


def test_training_pairs_only_reference_train_b_records(everything):
    """The hard-negative rule: no validation or test B record in training."""
    train_b_ids = set(everything["train_b"]["id"])
    assert set(everything["train_pairs"]["idB"]) <= train_b_ids


def test_training_pairs_only_reference_train_a_records(everything):
    train_a_ids = set(everything["train_a"]["id"])
    assert set(everything["train_pairs"]["idA"]) <= train_a_ids


def test_no_gold_pair_crosses_train_and_validation(everything):
    train_a_ids = set(everything["train_a"]["id"])
    val_a_ids = set(everything["val_a"]["id"])
    assert not (train_a_ids & val_a_ids)


def test_validation_candidates_include_the_full_b_catalog(everything):
    """PRIMARY protocol: the pool is all of Catalog B, including test records."""
    assert len(everything["buy"]) == 1092
    assert set(everything["val_candidates"]["idB"]) <= set(everything["buy"]["id"])


def test_retrieval_finds_the_gold_match_for_every_validation_record(everything):
    metrics = candidate_recall(
        everything["val_candidates"],
        everything["val_gold"],
        set(everything["val_a"]["id"]),
    )
    assert metrics["matchable_a_records"] == 161
    assert metrics["recall_at_20"] == 1.0


def test_training_pairs_are_mostly_negative_and_class_balance_is_recorded(everything):
    pairs = everything["train_pairs"]
    positives = int(pairs["label"].sum())
    assert positives > 0
    assert len(pairs) - positives > positives  # deliberately negative-heavy


def test_features_are_finite_on_every_validation_pair(everything):
    frame = build_feature_frame(
        everything["val_candidates"],
        everything["abt"],
        everything["buy"],
        everything["similarity_model"],
    )
    matrix = feature_matrix(frame)
    assert matrix.shape[0] == len(everything["val_candidates"])
    assert matrix.min() >= 0.0
    assert matrix.max() <= 1.0


def test_pipeline_trains_and_emits_at_most_one_match_per_a(everything):
    model = fit_pair_model(everything["train_pairs"])
    frame = build_feature_frame(
        everything["val_candidates"],
        everything["abt"],
        everything["buy"],
        everything["similarity_model"],
    )
    scores = model.score(frame)
    decisions = apply_decisions(frame, scores, 0.9, 0.5)
    matched = decisions[decisions["decision"] == "MATCH"]
    assert matched["idA"].is_unique
    assert set(decisions["decision"]) <= {"MATCH", "REVIEW", "NO MATCH"}


def test_metrics_json_exists_and_quarantines_the_test_split():
    with open("reports/metrics.json") as handle:
        metrics = json.load(handle)
    assert metrics["test_split"]["status"] == "QUARANTINED"
    assert metrics["candidate_pool_protocol"].startswith("PRIMARY")


def test_metrics_json_reports_retrieval_and_matching_separately():
    with open("reports/metrics.json") as handle:
        metrics = json.load(handle)
    for key in (
        "recall_at_5",
        "recall_at_10",
        "recall_at_20",
        "reduction_ratio",
        "remaining_candidate_fraction",
    ):
        assert key in metrics["retrieval"]
    for key in ("precision", "recall", "f1", "false_merges"):
        assert key in metrics["pair_level_validation"]
    for key in ("hits_at_1", "hits_at_5", "hits_at_10", "mrr"):
        assert key in metrics["product_level_validation"]


def test_reduction_conventions_are_complementary_in_the_report():
    with open("reports/metrics.json") as handle:
        metrics = json.load(handle)
    retrieval = metrics["retrieval"]
    assert (
        retrieval["remaining_candidate_fraction"] + retrieval["reduction_ratio"]
        == pytest.approx(1.0)
    )


def test_product_level_is_ranked_by_matcher_score_not_retrieval():
    """Regression guard: the reported product-level metrics must use model score."""
    with open("reports/metrics.json") as handle:
        metrics = json.load(handle)
    assert metrics["product_level_validation"]["ranked_by"] == "score"
    assert metrics["product_level_by_retrieval_rank"]["ranked_by"] == "rank"


def test_gold_coverage_and_threshold_diagnostic_are_reported():
    with open("reports/metrics.json") as handle:
        metrics = json.load(handle)
    coverage = metrics["gold_coverage_by_decision_validation"]
    assert coverage["evaluated_gold_pairs"] == 166
    assert (
        coverage["gold_pairs_as_MATCH"]
        + coverage["gold_pairs_as_REVIEW"]
        + coverage["gold_pairs_as_NO_MATCH"]
    ) == coverage["evaluated_gold_pairs"]
    diagnostic = metrics["threshold_diagnostic_validation"]
    assert [row["threshold"] for row in diagnostic] == sorted(
        row["threshold"] for row in diagnostic
    )


def test_error_analysis_csv_uses_real_benchmark_rows():
    frame = pd.read_csv("reports/error_analysis.csv")
    assert len(frame) > 0
    assert {"error_type", "idA", "idB", "abt_name", "buy_name"} <= set(frame.columns)
  # Real names, not placeholders.
    assert frame["abt_name"].str.len().min() > 3
    assert set(frame["error_type"]) <= {
        "false_merge",
        "gold_not_ranked_first",
        "correct_but_sent_to_review",
        "rejected_despite_being_top",
        "gold_never_retrieved",
    }


def test_no_retrieval_failure_is_claimed_when_recall_at_20_is_perfect():
    """Regression guard for a misleading label.

    An earlier version labelled any product whose retrieval-top-1 was not gold as
    "retrieval_failure". With candidate Recall@20 == 1.0 there are no retrieval
    failures at all, so that label hid matcher errors behind a retrieval excuse.
    """
    with open("reports/metrics.json") as handle:
        metrics = json.load(handle)
    frame = pd.read_csv("reports/error_analysis.csv")
    if metrics["retrieval"]["recall_at_20"] == 1.0:
        assert (frame["error_type"] == "gold_never_retrieved").sum() == 0
        assert "retrieval_failure" not in set(frame["error_type"])