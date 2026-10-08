"""Evaluation, kept separate from retrieval.

Retrieval (blocking.py) asks whether the right Buy record can be found at all.
Matching (here) asks whether a candidate pair is the same product. A gold pair
lost during blocking cannot be recovered by a better matcher, so the two are
reported side by side rather than collapsed into one score.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# 16 products have two gold partners and may only receive one B record, so at
# most 1081 of the 1097 gold pairs can ever be predicted positive.
PAIR_RECALL_CEILING = 1081 / 1097


def pair_level_metrics(gold, decisions, eval_a_ids):
    """Pair precision / recall / F1 over the evaluated candidate pairs.

    A predicted MATCH counts as a positive. A gold pair absent from the candidate
    list cannot count towards recall here; that loss is reported separately by
    candidate_recall.
    """
    evaluated = decisions[decisions["idA"].isin(eval_a_ids)]
    gold_pairs = set(zip(gold["idAbt"], gold["idBuy"]))

    labels, predictions = [], []
    for id_a, id_b, decision in zip(evaluated["idA"], evaluated["idB"], evaluated["decision"]):
        labels.append(int((id_a, id_b) in gold_pairs))
        predictions.append(int(decision == "MATCH"))

    labels = np.array(labels)
    predictions = np.array(predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])

    true_positive = int(matrix[1, 1])
    false_positive = int(matrix[0, 1])
    false_negative = int(matrix[1, 0])

    is_gold_pair = np.array(
        [(a, b) in gold_pairs for a, b in zip(evaluated["idA"], evaluated["idB"])]
    )
    is_match = (evaluated["decision"] == "MATCH").to_numpy()

    # A false merge is a MATCH predicted on two records that are NOT the same
    # product. In a real catalog this silently corrupts one of the records.
    false_merges = evaluated[is_match & ~is_gold_pair]
    # A false non-match is a gold pair the system refused to merge.
    false_non_matches = evaluated[~is_match & is_gold_pair]

    return {
        "evaluated_candidate_pairs": int(len(evaluated)),
        "gold_pairs_in_evaluated_set": int(labels.sum()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "confusion_matrix": matrix.tolist(),
        "false_merges": int(len(false_merges)),
        "false_non_matches": int(len(false_non_matches)),
        "pair_recall_ceiling": PAIR_RECALL_CEILING,
    }


def product_level_metrics(ranked, gold, eval_a_ids, order_by="rank"):
    """Hits@1/5/10 and MRR for products that have at least one gold match.

    order_by is "rank" (retrieval order) or "score" (model order). They answer
    different questions: ranking by retrieval reproduces candidate Recall@K by
    construction and says nothing about the matcher, while ranking by model score
    asks whether the matcher put the right record first among the candidates it
    was given. The score ranking is the real product-level measure.

    A hit means SOME gold record is in the top K, so this is not capped by the
    one-MATCH-per-A rule the way pair-level recall is.
    """
    if order_by not in ranked.columns:
        raise KeyError(f"ranking column {order_by!r} not in {list(ranked.columns)}")
    gold_pairs = set(zip(gold["idAbt"], gold["idBuy"]))
    gold_by_a: dict[str, set[str]] = {}
    for id_a, id_b in gold_pairs:
        gold_by_a.setdefault(id_a, set()).add(id_b)

    matchable = [id_a for id_a in sorted(eval_a_ids) if id_a in gold_by_a]

    hits = {1: 0, 5: 0, 10: 0}
    reciprocal_ranks = []
    for id_a in matchable:
        rows = ranked[ranked["idA"] == id_a]
        # Retrieval rank 1 is the BEST candidate, so it sorts ascending.
        # Model score is highest-is-best, so it sorts descending.
        ascending = order_by == "rank"
        rows = rows.sort_values(order_by, ascending=ascending, kind="stable")
        ordered = rows["idB"].tolist()
        gold_b = gold_by_a[id_a]
        first_hit = None
        for position, id_b in enumerate(ordered, start=1):
            if id_b in gold_b:
                first_hit = position
                break
        if first_hit is not None:
            reciprocal_ranks.append(1.0 / first_hit)
            for k in hits:
                if first_hit <= k:
                    hits[k] += 1

    denominator = len(matchable)
    return {
        "ranked_by": order_by,
        "matchable_a_records": denominator,
        "hits_at_1": hits[1] / denominator if denominator else 0.0,
        "hits_at_5": hits[5] / denominator if denominator else 0.0,
        "hits_at_10": hits[10] / denominator if denominator else 0.0,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "never_found": denominator - len(reciprocal_ranks),
    }


def threshold_diagnostic(
    frame: pd.DataFrame,
    scores: np.ndarray,
    gold_pairs: set[tuple[str, str]],
    review_threshold: float,
    thresholds: list[float],
) -> list[dict]:
    """Validation precision/recall/F1 across a threshold sweep.

    DIAGNOSTIC ONLY. This does not select the threshold and must not be used to
    select one. The registered rule in docs/methodology.md is unchanged; this
    sweep exists to explain what the rule's grid could and could not see.
    """
    from entity_resolution.model import apply_decisions

    rows = []
    for threshold in thresholds:
        decisions = apply_decisions(frame, scores, threshold, review_threshold)
        metrics = pair_level_metrics(
            pd.DataFrame(list(gold_pairs), columns=["idAbt", "idBuy"]),
            decisions,
            set(frame["idA"]),
        )
        rows.append(
            {
                "threshold": threshold,
                "match_pairs": metrics["true_positives"] + metrics["false_positives"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "false_merges": metrics["false_merges"],
            }
        )
    return rows


def gold_coverage_by_decision(
    gold: pd.DataFrame, decisions: pd.DataFrame, eval_a_ids: set[str]
) -> dict:
    """Where did the gold pairs end up, and what does that cost?

    Splits every evaluated gold pair by the decision it received, so the cost of
    a conservative MATCH threshold is visible instead of hidden inside a single
    recall number.

    recall_match_only   counts a gold pair recovered only if it was MATCHed.
    recall_match_review counts it recovered if it was MATCHed OR sent to REVIEW.
                         That second definition is meaningful because REVIEW is a
                         human-in-the-loop outcome, not an automated merge.
    """
    evaluated = decisions[decisions["idA"].isin(eval_a_ids)]
    gold_pairs = set(zip(gold["idAbt"], gold["idBuy"]))

    gold_rows = evaluated[
        [(a, b) in gold_pairs for a, b in zip(evaluated["idA"], evaluated["idB"])]
    ]
    by_decision = gold_rows["decision"].value_counts().to_dict()
    total_gold = len(gold_rows)

    matched = int(by_decision.get("MATCH", 0))
    review = int(by_decision.get("REVIEW", 0))
    no_match = int(by_decision.get("NO MATCH", 0))

    return {
        "evaluated_gold_pairs": total_gold,
        "gold_pairs_as_MATCH": matched,
        "gold_pairs_as_REVIEW": review,
        "gold_pairs_as_NO_MATCH": no_match,
        "recall_match_only": matched / total_gold if total_gold else 0.0,
        "recall_match_plus_review": (matched + review) / total_gold if total_gold else 0.0,
        "false_non_matches_sent_to_REVIEW": review,
        "false_non_matches_sent_to_NO_MATCH": no_match,
        "note": (
            "A gold pair in REVIEW was retrieved and not rejected; a human would "
            "still have to confirm it. It is NOT an automated match, so "
            "recall_match_only is the honest automated figure."
        ),
    }


ERROR_CATEGORIES = {
    "false_merge": "MATCH emitted on a pair that is not gold. The costly error.",
    "missed_match": (
        "The top candidate is gold but the decision was not MATCH, split into "
        "correct_but_sent_to_review and rejected_despite_being_top. Not a false "
        "negative in the usual sense: the system never claimed a non-match."
    ),
    "gold_not_ranked_first": (
        "A gold partner was retrieved but the matcher preferred another "
        "candidate. Retrieval worked, matching failed."
    ),
    "gold_never_retrieved": (
        "No gold partner is in the candidate list. The only true retrieval "
        "failure, and no matcher can recover from it."
    ),
}


def classify_outcome(top_id_b, decision, gold_b, gold_in_candidates):
    """One label per product.

    Note there is deliberately no "retrieval_failure" label for a case where
    retrieval did its job. Candidate Recall@20 is 1.0 here, so there are zero
    true retrieval failures; calling a mis-ranked candidate one would hide
    matcher errors behind a retrieval excuse.
    """
    if decision == "MATCH":
        return "false_merge" if top_id_b not in gold_b else "correct_match"

    top_is_gold = top_id_b in gold_b
    if not gold_in_candidates:
        # Nothing correct was even offered to the matcher.
        return "gold_never_retrieved"
    if top_is_gold:
        return "rejected_despite_being_top" if decision == "NO MATCH" else "correct_but_sent_to_review"
    return "gold_not_ranked_first"


def gold_partner_ranks(candidates, gold_b):
    """Retrieval ranks (1-based) at which this product's gold partners sit."""
    ordered = candidates.sort_values("rank")["idB"].tolist()
    return [i for i, value in enumerate(ordered, start=1) if value in gold_b]


def case_tags(row: pd.Series) -> str:
    """Labels describing why a pair is hard, computed from its real features."""
    tags = []
    if row["feature_model_token_conflict"] == 1.0:
        tags.append("model_number_conflict")
    if row["feature_manufacturer_in_name"] == 1.0:
        tags.append("same_brand_lookalike")
    elif str(row.get("buy_manufacturer", "")).strip():
        tags.append("manufacturer_conflict")
    if row["feature_price_known_both"] == 0.0:
        tags.append("price_missing_on_a_side")
    if row["feature_name_similarity"] < 0.4:
        tags.append("lexical_failure")
    if row["feature_description_similarity"] == 0.0:
        tags.append("description_missing")
    if row["feature_numeric_token_overlap"] == 0.0 and row["feature_model_token_conflict"] == 1.0:
        tags.append("conflicting_model_and_size")
    return "|".join(tags) if tags else "none"


def write_error_analysis(decisions, candidates, gold, abt, buy, eval_a_ids, feature_names):
    """One row per A record that went wrong, with the evidence behind it."""
    gold_pairs = set(zip(gold["idAbt"], gold["idBuy"]))
    a_lookup = abt.set_index("id")
    b_lookup = buy.set_index("id")

    records = []
    for id_a, group in decisions.groupby("idA"):
        if id_a not in eval_a_ids:
            continue
        gold_b = {b for a, b in gold_pairs if a == id_a}
        top = group.sort_values("score_rank").iloc[0]
        gold_ranks = gold_partner_ranks(candidates[candidates["idA"] == id_a], gold_b)

        error_type = classify_outcome(top["idB"], top["decision"], gold_b, bool(gold_ranks))
        if error_type == "correct_match":
            continue

        a_record = a_lookup.loc[id_a]
        b_record = b_lookup.loc[top["idB"]]
        records.append(
            {
                "error_type": error_type,
                "idA": id_a,
                "idB": top["idB"],
                "decision": top["decision"],
                "score": round(float(top["score"]), 6),
                "retrieval_rank": int(top["rank"]),
                "gold_in_candidate_list": bool(gold_ranks),
                "abt_name": a_record["name"],
                "abt_description": a_record["description"],
                "abt_price": a_record["price"],
                "buy_name": b_record["name"],
                "buy_description": b_record["description"],
                "buy_manufacturer": b_record.get("manufacturer", ""),
                "buy_price": b_record["price"],
                "gold_partners": "|".join(sorted(gold_b)),
                **{f"feature_{n}": round(float(top[n]), 4) for n in feature_names},
            }
        )

    frame = pd.DataFrame.from_records(records)
    if not frame.empty:
        frame["case_tags"] = frame.apply(case_tags, axis=1)
    return frame


def save_error_analysis(frame, split_name, path):
    """Merge one split's rows into the shared error_analysis.csv.

    Rows from the other split are kept, so re-running the training pipeline does
    not destroy the test rows.
    """
    keep = pd.DataFrame()
    if path.exists():
        existing = pd.read_csv(path)
        keep = existing[existing["split"] != split_name]

    frame = frame.copy()
    frame.insert(0, "split", split_name)
    combined = pd.concat([keep, frame], ignore_index=True)
    combined.sort_values(["split", "error_type", "idA"]).reset_index(drop=True)
    combined.to_csv(path, index=False)

    print(f"\n{split_name} error breakdown (documentation only):")
    print(frame["error_type"].value_counts().to_string())
    counts = {}
    for tags in frame["case_tags"]:
        for tag in tags.split("|"):
            counts[tag] = counts.get(tag, 0) + 1
    print(
        "case tags: "
        + ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
    )


def abstention_probe(candidates_without_gold, scores, match_threshold):
    """DIAGNOSTIC ONLY, not benchmark precision.

    Every A record in this benchmark has a gold match, so real NO MATCH
    performance cannot be measured. This removes an A record's own gold partner
    from its candidate pool and checks whether the best remaining candidate still
    clears the MATCH threshold. Never used for tuning.
    """
    if len(candidates_without_gold) == 0:
        return {"a_records_probed": 0, "abstained": 0}

    frame = candidates_without_gold.copy()
    frame["score"] = scores
    best_per_a = frame.sort_values(["idA", "score"], ascending=[True, False], kind="stable")
    best_per_a = best_per_a.groupby("idA", as_index=False).first()

    best_scores = best_per_a["score"].to_numpy()
    # The probe asks whether the system would still emit a MATCH when the true
    # partner is gone. match_threshold is passed in for that purpose, so the
    # result is reported as "still above the MATCH boundary".
    above = best_scores >= match_threshold
    return {
        "a_records_probed": int(len(best_per_a)),
        "still_above_match_boundary": int(above.sum()),
        "abstained": int((~above).sum()),
        "abstention_rate": float((~above).mean()),
    }


def end_to_end_metrics(
    gold: pd.DataFrame, retrieval: dict, pair_metrics: dict
) -> dict:
    """Combines retrieval loss with matcher performance.

    End-to-end recall multiplies the chance of finding the gold record by the
    chance of then matching it, so a matcher cannot hide behind retrieval.
    """
    return {
        "candidate_recall_at_20": retrieval["recall_at_20"],
        "matches_lost_in_blocking": retrieval["lost_in_blocking_at_20"],
        "pair_recall": pair_metrics["recall"],
        "end_to_end_recall": retrieval["recall_at_20"] * pair_metrics["recall"],
    }