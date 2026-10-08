"""Candidate retrieval (blocking).

Answers one question: can the correct Buy record be found among the candidates at
all? Kept separate from matching, because a true pair lost here cannot be
recovered by a better matcher.

Character n-gram TF-IDF over normalized names, cosine similarity, top-k. Character
n-grams rather than words because the same product is written very differently in
the two catalogs ("PSLX350H" vs "pslx 350h belt drive") and character fragments
survive that spelling drift where whole words do not.

Leakage control: the vectorizer is fitted on train A and train B names only.
Validation and test names are transformed with the frozen vectorizer.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as l2_normalize

from entity_resolution.normalize import normalize

TOP_K = 20


def _join_names(frame: pd.DataFrame) -> list[str]:
    """Normalized 'name' for every row of a catalog frame."""
    return [normalize(value) for value in frame["name"]]


def fit_retriever(train_a: pd.DataFrame, train_b: pd.DataFrame) -> TfidfVectorizer:
    """Fit the char n-gram TF-IDF on TRAIN names only.

    train_a and train_b must contain TRAIN-split records only. The returned
    vectorizer has already seen no validation or test text.
    """
    texts = _join_names(train_a) + _join_names(train_b)
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 4),
        min_df=1,
        sublinear_tf=True,
        lowercase=False,  # normalize() already lowercased
    )
    vectorizer.fit(texts)
    return vectorizer


def retrieve(
    vectorizer: TfidfVectorizer,
    query_a: pd.DataFrame,
    catalog_b: pd.DataFrame,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """Return the top_k nearest B records for every A record.

    Output columns: idA, rank, idB, cosine_sim.
    rank 1 is the highest-scoring candidate. Similarities are L2-normalized
    dot products, so they are cosine similarities in [0, 1].
    """
    query_matrix = l2_normalize(vectorizer.transform(_join_names(query_a)))
    catalog_matrix = l2_normalize(vectorizer.transform(_join_names(catalog_b)))

    # Similarity of every A record against every B record. The product of two
    # sparse matrices is still sparse, so it is densified: the grid here is
    # A x B (about 1081 x 1092), roughly 1.2 million floats.
    similarity = (query_matrix @ catalog_matrix.T).toarray()

    k = min(top_k, catalog_b.shape[0])
    b_ids = catalog_b["id"].to_numpy()
    a_ids = query_a["id"].to_numpy()

    # argpartition finds the top-k cheaply; then sort just those k.
    top_indices = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(len(a_ids))[:, None]
    top_scores = similarity[rows, top_indices]
    order = np.argsort(-top_scores, axis=1, kind="stable")

    records = []
    for row in range(len(a_ids)):
        for position, column in enumerate(order[row], start=1):
            b_index = top_indices[row, column]
            records.append(
                {
                    "idA": a_ids[row],
                    "rank": position,
                    "idB": b_ids[b_index],
                    "cosine_sim": float(top_scores[row, column]),
                }
            )
    return pd.DataFrame.from_records(
        records, columns=["idA", "rank", "idB", "cosine_sim"]
    )


def candidate_recall(
    candidates: pd.DataFrame, gold: pd.DataFrame, query_ids: set[str]
) -> dict[str, float]:
    """Recall@K over A records in query_ids that have at least one gold match.

    Denominator is A records with a gold match, not pairs. The
    one-MATCH-per-A policy caps pair-level recall at 1081/1097 = 98.54%, but
    retrieval has no such cap: a hit means any gold B record is in the top K.
    """
    gold_pairs = set(zip(gold["idAbt"], gold["idBuy"]))
    matchable = {
        id_a for id_a, _ in gold_pairs if id_a in query_ids
    }
    hits = {k: 0 for k in (5, 10, 20)}
    for id_a, group in candidates.groupby("idA"):
        if id_a not in matchable:
            continue
        gold_b = {b for a, b in gold_pairs if a == id_a}
        ranked = group.sort_values("rank")["idB"].tolist()
        for k in hits:
            if gold_b & set(ranked[:k]):
                hits[k] += 1
    denominator = len(matchable)
    return {
        "recall_at_5": hits[5] / denominator,
        "recall_at_10": hits[10] / denominator,
        "recall_at_20": hits[20] / denominator,
        "matchable_a_records": denominator,
        "lost_in_blocking_at_20": denominator - hits[20],
    }


def remaining_candidate_fraction(
    candidates: pd.DataFrame, query_a: pd.DataFrame, catalog_b: pd.DataFrame
) -> float:
    """Fraction of the full A x B grid that blocking leaves for the matcher."""
    full = len(query_a) * len(catalog_b)
    return len(candidates) / full if full else 0.0


def reduction_ratio(candidates: pd.DataFrame, query_a: pd.DataFrame, catalog_b: pd.DataFrame) -> float:
    """Conventional reduction ratio = 1 - remaining fraction, in [0, 1].

    Reported alongside remaining_candidate_fraction because the two conventions
    are both in common use and quoting only one is ambiguous:
      remaining fraction  = 0.0183  (1.83% of the grid survives blocking)
      reduction ratio     = 0.9817  (98.17% of the grid was removed)
    """
    return 1.0 - remaining_candidate_fraction(candidates, query_a, catalog_b)