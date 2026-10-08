"""Logistic Regression matcher and the MATCH / REVIEW / NO MATCH decision policy.

Decision rules, fixed before the test split was touched:
- candidates for an A record are ranked by model score
- only the top ranked candidate may become MATCH, and only above the MATCH
  threshold, so one A record can never be merged twice
- a candidate between the two thresholds becomes REVIEW
- everything else is NO MATCH, and an A record is never forced to match

MATCH is the lowest validation threshold reaching 0.95 precision. REVIEW is the
95th percentile of TRAIN negative scores. See docs/methodology.md.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from entity_resolution.features import FEATURE_NAMES

MATCH_PRECISION_TARGET = 0.95
REVIEW_PERCENTILE = 95.0

DECISIONS = ["MATCH", "REVIEW", "NO MATCH"]


@dataclass
class PairModel:
    """Fitted Logistic Regression plus the thresholds decided on validation."""
    pipeline: Pipeline
    feature_names: list = field(default_factory=lambda: list(FEATURE_NAMES))
    train_positive_count: int = 0
    train_negative_count: int = 0
    match_threshold: float = 0.5
    review_threshold: float = 0.5

    def score(self, frame):
        """Estimated probability that a pair is the same product.

        Not called a confidence because calibration was never evaluated.
        """
        columns = frame[self.feature_names].to_numpy(dtype=float)
        return self.pipeline.predict_proba(columns)[:, 1]

    def coefficients(self):
        table = self.pipeline.named_steps["logisticregression"].coef_[0]
        return pd.DataFrame({"feature": self.feature_names, "coefficient": table}).sort_values(
            "coefficient", ascending=False
        )


def fit_pair_model(frame, feature_names=None):
    """Fit on labelled TRAIN candidate pairs.

    The scaler sits inside the Pipeline so it is fitted on the training rows only.
    """
    names = list(feature_names) if feature_names else list(FEATURE_NAMES)
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logisticregression",
                LogisticRegression(
                    class_weight="balanced", max_iter=2000, random_state=42
                ),
            ),
        ]
    )
    pipeline.fit(frame[names].to_numpy(dtype=float), frame["label"].to_numpy())

    labels = frame["label"].to_numpy()
    return PairModel(
        pipeline=pipeline,
        feature_names=names,
        train_positive_count=int(labels.sum()),
        train_negative_count=int((labels == 0).sum()),
    )


def apply_decisions(frame, scores, match_threshold, review_threshold):
    """Add score, score_rank and decision columns."""
    result = frame.copy()
    result["score"] = np.asarray(scores, dtype=float)
    result = result.sort_values(
        ["idA", "score"], ascending=[True, False], kind="stable"
    )
    result["score_rank"] = result.groupby("idA").cumcount() + 1

    result["decision"] = "NO MATCH"
    result.loc[result["score"] >= review_threshold, "decision"] = "REVIEW"
    match = (result["score"] >= match_threshold) & (result["score_rank"] == 1)
    result.loc[match, "decision"] = "MATCH"
    return result.sort_values(["idA", "score_rank"]).reset_index(drop=True)


def summarise_queue(decisions):
    counts = decisions["decision"].value_counts().to_dict()
    total = len(decisions)
    review = counts.get("REVIEW", 0)
    return {
        "candidate_pairs": total,
        "match_pairs": counts.get("MATCH", 0),
        "review_pairs": review,
        "no_match_pairs": counts.get("NO MATCH", 0),
        "review_queue_size": review,
        "review_queue_rate": review / total if total else 0.0,
        "a_records": int(decisions["idA"].nunique()),
        "a_records_matched": int(
            decisions[decisions["decision"] == "MATCH"]["idA"].nunique()
        ),
    }


def select_match_threshold(frame, scores, gold_pairs, review_threshold):
    """Lowest achievable MATCH threshold whose validation precision reaches 0.95.

    Returns (threshold, precision_at_that_threshold, target_reached).

    The candidates are the sorted unique observed validation scores rather than a
    fixed grid. A decision uses score >= threshold, so the MATCH set only changes
    when the threshold crosses an observed score; every other value reproduces one
    of these. A grid capped at 0.95 would make the rule unable to satisfy its own
    0.95 target, which is what an earlier version of this did.

    Precision is measured on the MATCH decisions the system actually emits, so
    the full decision policy runs at each threshold. Counting every pair above the
    threshold instead would measure a different operating point from the one
    that gets reported.
    """
    candidates = sorted({round(float(value), 12) for value in np.asarray(scores)})

    best_threshold, best_precision = 0.0, 0.0
    for threshold in candidates:
        decisions = apply_decisions(frame, scores, threshold, review_threshold)
        matched = decisions[decisions["decision"] == "MATCH"]
        if matched.empty:
            continue
        correct = sum(
            1 for a, b in zip(matched["idA"], matched["idB"]) if (a, b) in gold_pairs
        )
        precision = correct / len(matched)
        if precision >= MATCH_PRECISION_TARGET:
            return threshold, float(precision), True
        if precision > best_precision:
            best_threshold, best_precision = threshold, float(precision)
    return best_threshold, best_precision, False


def set_review_threshold(model, train_negative_scores):
    """REVIEW boundary = 95th percentile of TRAIN negative-pair scores."""
    if len(train_negative_scores) == 0:
        model.review_threshold = model.match_threshold
    else:
        model.review_threshold = float(
            np.percentile(train_negative_scores, REVIEW_PERCENTILE)
        )