"""Final test evaluation. Run ONCE: python -m entity_resolution.evaluate_test

Separate from train.py so the training pipeline stays untouched by the test run.
Nothing is fitted, tuned or re-selected here: the thresholds are recorded in
docs/methodology.md and asserted against the saved model, so a mismatch fails
loudly instead of silently re-tuning.

Test records and gold labels are read exactly once, in this module. Test results
are for reporting only; nothing downstream may modify the model because of them.
"""
import json

import numpy as np
import pandas as pd
from joblib import load

from entity_resolution.blocking import (
    candidate_recall,
    fit_retriever,
    reduction_ratio,
    remaining_candidate_fraction,
    retrieve,
)
from entity_resolution.data import (
    PROJECT_ROOT,
    gold_in_split,
    load_splits,
    records_in_split,
    split_records,
)
from entity_resolution.evaluation import (
    abstention_probe,
    end_to_end_metrics,
    gold_coverage_by_decision,
    pair_level_metrics,
    product_level_metrics,
    write_error_analysis,
    save_error_analysis,
)
from entity_resolution.features import FEATURE_NAMES, build_feature_frame
from entity_resolution.model import apply_decisions, summarise_queue
from entity_resolution.features import fit_similarity

TOP_K = 20
REPORT_DIR = PROJECT_ROOT / "reports"

# Frozen on validation, before the test split was read. Never re-derived here.
FROZEN_MATCH_THRESHOLD = 0.9856394449
FROZEN_REVIEW_THRESHOLD = 0.5182070609145072
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 42


def main() -> None:
    abt, buy, gold = split_records()
    splits = load_splits()

    train_a = records_in_split(abt, splits, "train")
    train_b = records_in_split(buy, splits, "train")

    # ---- 1. refit the train-only transforms, deterministically ----
    retriever = fit_retriever(train_a, train_b)
    similarity_model = fit_similarity(train_a, train_b)

    # ---- 2. load the FROZEN fitted matcher ----
    bundle = load(PROJECT_ROOT / "models" / "matcher.joblib")
    model = bundle["model"]

    # Assert the saved model matches the frozen decisions. A mismatch means the
    # artefacts are stale, and failing loudly beats quietly re-tuning.
    if abs(model.match_threshold - FROZEN_MATCH_THRESHOLD) > 1e-12:
        raise RuntimeError(
            f"saved MATCH threshold {model.match_threshold} != frozen "
            f"{FROZEN_MATCH_THRESHOLD}. Artefacts are stale."
        )
    if abs(model.review_threshold - FROZEN_REVIEW_THRESHOLD) > 1e-9:
        raise RuntimeError(
            f"saved REVIEW threshold {model.review_threshold} != frozen "
            f"{FROZEN_REVIEW_THRESHOLD}. Artefacts are stale."
        )

    # ---- 3. TEST: retrieved here for the first and only time ----
    test_a = records_in_split(abt, splits, "test")
    test_gold = gold_in_split(gold, abt, splits, "test")
    test_a_ids = set(test_a["id"])
    print(f"TEST A={len(test_a)} TEST gold pairs={len(test_gold)}")

    candidates = retrieve(retriever, test_a, buy, top_k=TOP_K)

    retrieval_metrics = candidate_recall(candidates, test_gold, test_a_ids)
    retrieval_metrics["remaining_candidate_fraction"] = remaining_candidate_fraction(
        candidates, test_a, buy
    )
    retrieval_metrics["reduction_ratio"] = reduction_ratio(candidates, test_a, buy)
    retrieval_metrics["candidate_pool_protocol"] = "PRIMARY: all Catalog B records"
    retrieval_metrics["top_k"] = TOP_K
    retrieval_metrics["matcher_sees_pairs"] = len(candidates)
    retrieval_metrics["full_grid_pairs"] = len(test_a) * len(buy)

    # ---- 4. score and decide with the frozen thresholds ----
    features = build_feature_frame(candidates, abt, buy, similarity_model)
    scores = model.score(features)
    decisions = apply_decisions(
        features, scores, FROZEN_MATCH_THRESHOLD, FROZEN_REVIEW_THRESHOLD
    )

    pair_metrics = pair_level_metrics(test_gold, decisions, test_a_ids)
    product_metrics = product_level_metrics(decisions, test_gold, test_a_ids, "score")
    retrieval_ranking = product_level_metrics(candidates, test_gold, test_a_ids, "rank")
    queue = summarise_queue(decisions)
    coverage = gold_coverage_by_decision(test_gold, decisions, test_a_ids)

    matched = decisions[decisions["decision"] == "MATCH"]
    assert matched["idA"].is_unique, "one-MATCH-per-A violated"

    print(
        f"retrieval R@5={retrieval_metrics['recall_at_5']:.4f} "
        f"R@10={retrieval_metrics['recall_at_10']:.4f} "
        f"R@20={retrieval_metrics['recall_at_20']:.4f} "
        f"remaining={retrieval_metrics['remaining_candidate_fraction']:.6f} "
        f"reduction={retrieval_metrics['reduction_ratio']:.6f}"
    )
    print(
        f"pair P={pair_metrics['precision']:.4f} R={pair_metrics['recall']:.4f} "
        f"F1={pair_metrics['f1']:.4f} macroF1={pair_metrics['macro_f1']:.4f} "
        f"FM={pair_metrics['false_merges']} FNM={pair_metrics['false_non_matches']}"
    )
    print(
        f"product (model rank) Hits@1={product_metrics['hits_at_1']:.4f} "
        f"Hits@5={product_metrics['hits_at_5']:.4f} "
        f"Hits@10={product_metrics['hits_at_10']:.4f} "
        f"MRR={product_metrics['mrr']:.4f}"
    )

    # ---- 5. abstention probe, diagnostic only ----
    gold_by_a: dict[str, set[str]] = {}
    for id_a, id_b in zip(test_gold["idAbt"], test_gold["idBuy"]):
        gold_by_a.setdefault(id_a, set()).add(id_b)
    keep = [
        not (row.idA in gold_by_a and row.idB in gold_by_a[row.idA])
        for row in candidates.itertuples(index=False)
    ]
    ablated = candidates[keep].reset_index(drop=True)
    ablated_features = build_feature_frame(ablated, abt, buy, similarity_model)
    probe = abstention_probe(
        ablated_features, model.score(ablated_features), FROZEN_MATCH_THRESHOLD
    )

    # ---- 6. optional bootstrap CIs, resampled by A product ----
    bootstrap = bootstrap_by_a_product(
        decisions, test_gold, test_a_ids, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED
    )

    metrics = {
        "test_split_evaluation": {
            "run_count": 1,
            "note": (
                "The TEST split was read once, here, after every modelling "
                "decision was frozen. No test result was used to change the "
                "split, retrieval, features, model or thresholds."
            ),
        },
        "frozen": {
            "match_threshold": FROZEN_MATCH_THRESHOLD,
            "review_threshold": FROZEN_REVIEW_THRESHOLD,
            "one_match_per_a": True,
            "candidate_pool_protocol": "PRIMARY: all Catalog B records",
            "retrieval": "char_wb (3,4) TF-IDF fitted on TRAIN A + TRAIN B names",
            "model": "Pipeline(StandardScaler, LogisticRegression(class_weight=balanced))",
        },
        "retrieval": retrieval_metrics,
        "pair_level_test": pair_metrics,
        "product_level_test_model_ranking": product_metrics,
        "product_level_test_retrieval_ranking": retrieval_ranking,
        "queue_test": queue,
        "gold_coverage_by_decision_test": coverage,
        "end_to_end_test": end_to_end_metrics(test_gold, retrieval_metrics, pair_metrics),
        "bootstrap_95ci_by_a_product": bootstrap,
        "abstention_probe_test_diagnostic_only": probe,
        "abstention_probe_caveat": (
            "Every A record in Abt-Buy has at least one gold match. The benchmark "
            "contains NO true unmatched A product, so real-world NO MATCH "
            "performance CANNOT be measured and is NOT claimed. This probe removes "
            "the gold partner from the pool as a diagnostic only."
        ),
    }

    report_dir = PROJECT_ROOT / "reports"
    (report_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    save_error_analysis(
        write_error_analysis(decisions, candidates, test_gold, abt, buy,
                             test_a_ids, FEATURE_NAMES),
        "test",
        REPORT_DIR / "error_analysis.csv",
    )

    print(f"\nwrote {report_dir / 'test_metrics.json'}")
    print(f"wrote {report_dir / 'error_analysis.csv'} (test rows)")


def bootstrap_by_a_product(
    decisions: pd.DataFrame,
    gold: pd.DataFrame,
    eval_a_ids: set[str],
    resamples: int,
    seed: int,
) -> dict:
    """95% bootstrap CIs resampled BY A PRODUCT, not by pair.

    Resampling pairs would treat one product's 20 candidates and its single
    outcome as independent, which understates the interval.
    """
    gold_pairs = set(zip(gold["idAbt"], gold["idBuy"]))
    gold_by_a: dict[str, set] = {}
    for id_a, id_b in gold_pairs:
        gold_by_a.setdefault(id_a, set()).add(id_b)

    # One pass: per product, did we predict MATCH, was the top candidate gold,
    # did the gold partner rank first, and how many gold pairs does it have.
    per_product = {}
    for id_a, group in decisions.groupby("idA"):
        if id_a not in eval_a_ids:
            continue
        top = group.sort_values("score_rank").iloc[0]
        ordered = group.sort_values("rank")["idB"].tolist()
        gold_b = gold_by_a.get(id_a, set())
        hit_at_1 = bool(ordered) and ordered[0] in gold_b
        per_product[id_a] = (
            top["decision"] == "MATCH",
            (top["idA"], top["idB"]) in gold_pairs,
            hit_at_1,
            len(gold_b),
        )

    ids = list(per_product)
    n = len(ids)
    rng = np.random.default_rng(seed)
    samples = {"precision": [], "recall": [], "f1": [], "hits_at_1": []}

    for _ in range(resamples):
        picked = rng.choice(ids, size=n, replace=True)
        true_pos = false_pos = hit1 = total_gold = 0
        for id_a in picked:
            predicted, top_is_gold, at_1, gold_count = per_product[id_a]
            # Every gold pair of the drawn product counts towards the recall
            # denominator, matched or not.
            total_gold += gold_count
            if predicted:
                if top_is_gold:
                    true_pos += 1
                else:
                    false_pos += 1
            if at_1:
                hit1 += 1
        predicted_pos = true_pos + false_pos
        precision = true_pos / predicted_pos if predicted_pos else 0.0
        recall = true_pos / total_gold if total_gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        samples["precision"].append(precision)
        samples["recall"].append(recall)
        samples["f1"].append(f1)
        samples["hits_at_1"].append(hit1 / n)

    def interval(values: list[float]) -> dict:
        array = np.array(values)
        return {
            "lower_2_5": float(np.percentile(array, 2.5)),
            "upper_97_5": float(np.percentile(array, 97.5)),
        }

    return {
        "method": "percentile bootstrap, resampled by Catalog A product",
        "resamples": resamples,
        "seed": seed,
        "units": n,
        "note": (
            "Intervals reflect sampling variation across A products only. With "
            f"{n} test products they are wide; treat them as a rough guide."
        ),
        "precision": interval(samples["precision"]),
        "recall": interval(samples["recall"]),
        "f1": interval(samples["f1"]),
        "hits_at_1": interval(samples["hits_at_1"]),
    }


if __name__ == "__main__":
    main()