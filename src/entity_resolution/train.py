"""Single reproducible pipeline for the traditional matcher.

    python -m entity_resolution.train

Trains on TRAIN and evaluates on VALIDATION. The TEST split is never loaded here;
see evaluate_test.py, which is run once and reads the saved matcher.
Writes reports/metrics.json and the validation rows of reports/error_analysis.csv.
"""
import json

import pandas as pd
from joblib import dump

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
    SPLIT_DIR,
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
    threshold_diagnostic,
    write_error_analysis,
    save_error_analysis,
)
from entity_resolution.features import FEATURE_NAMES, build_feature_frame
from entity_resolution.normalize import normalize
from entity_resolution.model import (
    MATCH_PRECISION_TARGET,
    fit_pair_model,
    select_match_threshold,
    set_review_threshold,
)
from entity_resolution.pairs import mine_training_pairs
from entity_resolution.model import apply_decisions, summarise_queue
from entity_resolution.features import fit_similarity

TOP_K = 20
CANDIDATE_POOL = "PRIMARY: all Catalog B records"
# Diagnostic sweep only. It does not and must not select the threshold; the
# corrected search in model.select_match_threshold does that over every unique
# validation score.
DIAGNOSTIC_THRESHOLDS = [0.05, 0.2, 0.4, 0.6, 0.75, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999]
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"


def main() -> None:
    abt, buy, gold = split_records()
    if not SPLIT_DIR.exists():
        raise SystemExit(
            f"No split files in {SPLIT_DIR}. They are gitignored, so generate "
            "them once first:\n  python -m entity_resolution.data"
        )
    splits = load_splits()

    train_a = records_in_split(abt, splits, "train")
    train_b = records_in_split(buy, splits, "train")
    val_a = records_in_split(abt, splits, "val")
    val_gold = gold_in_split(gold, abt, splits, "val")

    print(f"train A={len(train_a)} train B={len(train_b)} val A={len(val_a)}")

    retriever = fit_retriever(train_a, train_b)
    train_candidates = retrieve(retriever, train_a, train_b, top_k=TOP_K)
    val_candidates = retrieve(retriever, val_a, buy, top_k=TOP_K)

    val_a_ids = set(val_a["id"])
    retrieval_metrics = candidate_recall(val_candidates, val_gold, val_a_ids)
    retrieval_metrics["remaining_candidate_fraction"] = remaining_candidate_fraction(
        val_candidates, val_a, buy
    )
    retrieval_metrics["reduction_ratio"] = reduction_ratio(val_candidates, val_a, buy)
    retrieval_metrics["candidate_pool_protocol"] = CANDIDATE_POOL
    retrieval_metrics["top_k"] = TOP_K
    retrieval_metrics["matcher_sees_pairs"] = len(val_candidates)
    retrieval_metrics["full_grid_pairs"] = len(val_a) * len(buy)
    print(
        f"retrieval R@5={retrieval_metrics['recall_at_5']:.4f} "
        f"R@10={retrieval_metrics['recall_at_10']:.4f} "
        f"R@20={retrieval_metrics['recall_at_20']:.4f} "
        f"reduction={retrieval_metrics['reduction_ratio']:.4f}"
    )

    similarity_model = fit_similarity(train_a, train_b)
    train_pairs = mine_training_pairs(train_candidates, train_a, train_b, gold, similarity_model)
    model = fit_pair_model(train_pairs)
    class_balance = model.train_positive_count / max(
        model.train_positive_count + model.train_negative_count, 1
    )
    print(
        f"training pairs: {model.train_positive_count} positive, "
        f"{model.train_negative_count} negative, positive rate {class_balance:.4f}"
    )

    # Leakage guard: training pairs may only reference train B records.
    train_b_ids = set(train_b["id"])
    leaked = set(train_pairs["idB"]) - train_b_ids
    if leaked:
        raise RuntimeError(f"training pairs reference non-train B records: {len(leaked)}")

    val_features = build_feature_frame(val_candidates, abt, buy, similarity_model)
    val_scores = model.score(val_features)

    val_gold_pairs = set(zip(val_gold["idAbt"], val_gold["idBuy"]))

    # REVIEW comes from train negatives only, so it is fixed before validation.
    train_negative_scores = model.score(train_pairs[train_pairs["label"] == 0])
    set_review_threshold(model, train_negative_scores)

    # MATCH comes from the validation rule in model.py.
    threshold, precision_at_threshold, target_reached = select_match_threshold(
        val_features, val_scores, val_gold_pairs, model.review_threshold
    )
    model.match_threshold = threshold
    print(
        f"MATCH threshold={threshold:.10f} (val precision {precision_at_threshold:.4f}, "
        f"target {MATCH_PRECISION_TARGET} reached={target_reached}) "
        f"REVIEW threshold={model.review_threshold:.4f} (train negative p95)"
    )

    val_decisions = apply_decisions(
        val_features, val_scores, model.match_threshold, model.review_threshold
    )
    queue = summarise_queue(val_decisions)
    pair_metrics = pair_level_metrics(val_gold, val_decisions, val_a_ids)

    # Hits@K under both rankings. The matcher ranking is the real product-level
    # measure; the retrieval ranking reproduces candidate Recall@K by
    # construction, so reporting only one of them would hide the difference.
    retrieval_ranking = product_level_metrics(val_candidates, val_gold, val_a_ids, "rank")
    matcher_ranking = product_level_metrics(
        val_decisions, val_gold, val_a_ids, "score"
    )
    product_metrics = matcher_ranking
    print(
        f"pair P={pair_metrics['precision']:.4f} R={pair_metrics['recall']:.4f} "
        f"F1={pair_metrics['f1']:.4f}"
    )
    print(
        f"  product-level by MODEL score : Hits@1={matcher_ranking['hits_at_1']:.4f} "
        f"Hits@5={matcher_ranking['hits_at_5']:.4f} Hits@10={matcher_ranking['hits_at_10']:.4f} "
        f"MRR={matcher_ranking['mrr']:.4f}"
    )
    print(
        f"  product-level by RETRIEVAL   : Hits@1={retrieval_ranking['hits_at_1']:.4f} "
        f"Hits@5={retrieval_ranking['hits_at_5']:.4f} Hits@10={retrieval_ranking['hits_at_10']:.4f} "
        f"MRR={retrieval_ranking['mrr']:.4f}"
    )

    # Diagnostic only. The threshold above was chosen by model.py.
    diagnostics = threshold_diagnostic(
        val_features,
        val_scores,
        val_gold_pairs,
        model.review_threshold,
        DIAGNOSTIC_THRESHOLDS,
    )
    coverage = gold_coverage_by_decision(val_gold, val_decisions, val_a_ids)
    print("\nthreshold diagnostic (does not select the threshold):")
    for row in diagnostics:
        marker = "  <- selected" if row["threshold"] == model.match_threshold else ""
        print(
            f"  t={row['threshold']:.3f} n_match={row['match_pairs']:3d} "
            f"P={row['precision']:.4f} R={row['recall']:.4f} F1={row['f1']:.4f}{marker}"
        )
    print(
        f"\ngold-pair coverage: MATCH={coverage['gold_pairs_as_MATCH']} "
        f"REVIEW={coverage['gold_pairs_as_REVIEW']} "
        f"NO MATCH={coverage['gold_pairs_as_NO_MATCH']} "
        f"(recall MATCH-only {coverage['recall_match_only']:.4f}, "
        f"MATCH+REVIEW {coverage['recall_match_plus_review']:.4f})"
    )

    # Abstention probe: diagnostic only, never used for tuning.
    gold_by_a = {}
    for id_a, id_b in zip(val_gold["idAbt"], val_gold["idBuy"]):
        gold_by_a.setdefault(id_a, set()).add(id_b)
    keep = [
        not (row.idA in gold_by_a and row.idB in gold_by_a[row.idA])
        for row in val_candidates.itertuples(index=False)
    ]
    ablated = val_candidates[keep].reset_index(drop=True)
    ablated_features = build_feature_frame(ablated, abt, buy, similarity_model)
    probe = abstention_probe(
        ablated_features, model.score(ablated_features), model.match_threshold
    )
    print(
        f"abstention probe (secondary): {probe['still_above_match_boundary']}"
        f"/{probe['a_records_probed']} A records still had a candidate above MATCH"
    )

    baselines = evaluate_baselines(abt, buy, val_candidates, val_gold, val_a_ids)

    metrics = {
        "candidate_pool_protocol": CANDIDATE_POOL,
        "thresholds": {
            "match_threshold": model.match_threshold,
            "match_precision_on_validation": precision_at_threshold,
            "precision_target": MATCH_PRECISION_TARGET,
            "target_reached": target_reached,
            "review_threshold": model.review_threshold,
            "review_rule": "95th percentile of TRAIN negative-pair scores",
            "one_match_per_a": True,
        },
        "retrieval": retrieval_metrics,
        "training": {
            "positive_pairs": model.train_positive_count,
            "negative_pairs": model.train_negative_count,
            "positive_rate": class_balance,
            "class_weight": "balanced",
            "imbalance_choice": (
                "class_weight='balanced' rather than resampling: it changes the "
                "threshold's meaning rather than the data, which keeps the mined "
                "hard negatives intact and the pipeline simpler."
            ),
            "negatives_mined_from": "TRAIN A x TRAIN B only",
        },
        "pair_level_validation": pair_metrics,
        "product_level_validation": product_metrics,
        "product_level_by_retrieval_rank": retrieval_ranking,
        "queue_validation": queue,
        "gold_coverage_by_decision_validation": coverage,
        "threshold_diagnostic_validation": diagnostics,
        "end_to_end_validation": end_to_end_metrics(val_gold, retrieval_metrics, pair_metrics),
        "baselines_validation": baselines,
        "abstention_probe_secondary": probe,
        "abstention_probe_note": (
            "Every Abt record has a gold match, so true NO MATCH performance "
            "cannot be measured here. This probe removes the gold partner from "
            "the pool. Diagnostic only; it was not used to tune anything."
        ),
        "coefficients": model.coefficients().to_dict(orient="records"),
        "feature_names": FEATURE_NAMES,
        "ablation_validation": run_ablation(
            train_pairs, val_features, val_gold_pairs, val_a_ids
        ),
        "test_split": {
            "status": "QUARANTINED",
            "note": "Not loaded or scored by this pipeline; see evaluate_test.py.",
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    save_error_analysis(
        write_error_analysis(val_decisions, val_candidates, val_gold, abt, buy,
                             val_a_ids, FEATURE_NAMES),
        "validation",
        REPORT_DIR / "error_analysis.csv",
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dump(
        {
            "model": model,
            "retriever": retriever,
            "similarity_model": similarity_model,
        },
        MODEL_DIR / "matcher.joblib",
    )

    print(f"\nwrote {REPORT_DIR / 'metrics.json'}")
    print(f"wrote {REPORT_DIR / 'error_analysis.csv'}")
    print(f"wrote {MODEL_DIR / 'matcher.joblib'}")


def run_ablation(
    train_pairs: pd.DataFrame,
    val_features: pd.DataFrame,
    val_gold_pairs: set,
    val_a_ids: set,
) -> dict:
    """One small ablation: remove the model-token feature group.

    Chosen because the validation errors are dominated by lookalike products from
    the same brand (LG LMV1680BK vs an unnamed LG 2.0 cu.ft. microwave, Weber
    S-310 vs E-310), which is exactly what the model-token features exist to
    catch. Each variant re-selects its threshold with the same pre-registered
    rule, so the comparison is like for like.
    """
    group = ["model_token_overlap", "model_token_conflict"]
    results = {}
    for label, dropped in (("full_model", []), ("without_model_token_group", group)):
        names = [c for c in FEATURE_NAMES if c not in dropped]

        model = fit_pair_model(train_pairs, feature_names=names)
        scores = model.score(val_features)
        threshold, precision, reached = select_match_threshold(
            val_features, scores, val_gold_pairs, 0.5
        )
        decisions = apply_decisions(val_features, scores, threshold, 0.5)
        metrics = pair_level_metrics(
            pd.DataFrame(list(val_gold_pairs), columns=["idAbt", "idBuy"]),
            decisions,
            val_a_ids,
        )
        results[label] = {
            "features_removed": dropped,
            "match_threshold": threshold,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "false_merges": metrics["false_merges"],
            "precision_target_reached": reached,
        }
    return results


def evaluate_baselines(
    abt: pd.DataFrame, buy: pd.DataFrame, candidates: pd.DataFrame, gold: pd.DataFrame,
    eval_a_ids: set[str],
) -> dict:
    """The two baselines, no model involved: exact name match, nearest candidate."""
    gold_by_a: dict[str, set] = {}
    for id_a, id_b in zip(gold["idAbt"], gold["idBuy"]):
        gold_by_a.setdefault(id_a, set()).add(id_b)

    abt_names = {i: normalize(n) for i, n in zip(abt["id"], abt["name"])}
    buy_names = {i: normalize(n) for i, n in zip(buy["id"], buy["name"])}
    ranked = candidates.sort_values(["idA", "rank"])

    # Baseline 1: exact normalized-name match.
    # Only a CANDIDATE whose Buy name equals the Abt name counts. Comparing the
    # gold partner's name against the candidates would be an oracle, not a baseline.
    exact_hits = 0
    top1_hits = 0
    for id_a, group in ranked.groupby("idA"):
        if id_a not in eval_a_ids:
            continue
        if any(buy_names.get(id_b) == abt_names.get(id_a) for id_b in group["idB"]):
            exact_hits += 1
        if group.iloc[0]["idB"] in gold_by_a.get(id_a, set()):
            top1_hits += 1

    denominator = len(eval_a_ids)
    return {
        "exact_normalized_name_match": {
            "a_records_matched": exact_hits,
            "a_records": denominator,
            "product_recall": exact_hits / denominator if denominator else 0.0,
        },
        "nearest_lexical_candidate": {
            "a_records_matched": top1_hits,
            "a_records": denominator,
            "hits_at_1": top1_hits / denominator if denominator else 0.0,
        },
    }


if __name__ == "__main__":
    main()
