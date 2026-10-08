"""Pair features: ten values computed for one candidate pair.

Each one exists for a specific reason:

  name_similarity        main lexical signal, char n-gram TF-IDF cosine
  description_similarity free-form marketing text agreement, word TF-IDF cosine
  token_overlap          Jaccard on word tokens; reads differently from the cosine
  char_similarity        sequence ratio, robust to reordering
  model_token_overlap    Jaccard on model tokens (PSLX350H). A 15 inch and a 17
                         inch monitor share nearly every word; this separates them
  model_token_conflict   both sides name a model and the models disagree
  numeric_token_overlap  agreement on capacities and sizes (2.0 cu.ft., 16000)
  manufacturer_in_name   Buy manufacturer appears in the Abt name; A has no
                         manufacturer column (see normalize.manufacturer_in_name)
  price_difference       relative price gap, 0 when either side is missing
  price_known_both       flag for "both prices exist". Without it a missing price
                         would be indistinguishable from two equal prices,
                         since price_difference returns 0 for both

Retrieval rank is deliberately NOT a feature: it already encodes the same lexical
evidence as name_similarity and would blur the retrieval/matching boundary.

Two fitted TF-IDF objects feed name_similarity and description_similarity. Both
are fitted on train A + train B rows only; validation and test rows are
transformed with the frozen objects, never fitted on.
"""
import difflib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize as l2_normalize

from entity_resolution.normalize import (
    manufacturer_in_name,
    model_tokens,
    normalize,
    numeric_tokens,
    parse_price,
    price_difference,
    tokenize,
)

FEATURE_NAMES = [
    "name_similarity",
    "description_similarity",
    "token_overlap",
    "char_similarity",
    "model_token_overlap",
    "model_token_conflict",
    "numeric_token_overlap",
    "manufacturer_in_name",
    "price_difference",
    "price_known_both",
]


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


@dataclass
class SimilarityModel:
    """Fitted TF-IDF objects for names and descriptions. Train data only."""

    name_vectorizer: TfidfVectorizer
    description_vectorizer: TfidfVectorizer

    def name_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        texts = [normalize(v) for v in frame["name"]]
        return l2_normalize(self.name_vectorizer.transform(texts))

    def description_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        texts = [normalize(v) for v in frame["description"]]
        return l2_normalize(self.description_vectorizer.transform(texts))


def fit_similarity(train_a: pd.DataFrame, train_b: pd.DataFrame) -> SimilarityModel:
    """Fit both vectorizers on TRAIN rows only.

    Names use char n-grams because the same product is spelled differently in the
    two catalogs. Descriptions use words because they are free-form prose where
    word units fit better.
    """
    name_texts = [normalize(v) for v in train_a["name"]] + [
        normalize(v) for v in train_b["name"]
    ]
    description_texts = [normalize(v) for v in train_a["description"]] + [
        normalize(v) for v in train_b["description"]
    ]

    name_vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 4), min_df=1, sublinear_tf=True, lowercase=False
    )
    name_vectorizer.fit(name_texts)

    description_vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=False
    )
    try:
        description_vectorizer.fit(description_texts)
    except ValueError:
        # min_df=2 needs every term twice. The real train split has thousands of
        # terms, but a tiny or all-empty-description corpus would fail to fit.
        description_vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True, lowercase=False
        )
        description_vectorizer.fit(description_texts)

    return SimilarityModel(name_vectorizer, description_vectorizer)


def name_similarity(model, a_frame: pd.DataFrame, b_frame: pd.DataFrame) -> np.ndarray:
    """Dense cosine similarity between the names of two aligned frames."""
    return cosine_similarity(model.name_matrix(a_frame), model.name_matrix(b_frame))


def description_similarity(model, a_frame: pd.DataFrame, b_frame: pd.DataFrame) -> np.ndarray:
    """Dense cosine similarity between the descriptions of two aligned frames.

    Returns zeros when either side has no description text, because an empty
    vector has no direction to compare.
    """
    a_matrix = model.description_matrix(a_frame)
    b_matrix = model.description_matrix(b_frame)
    if a_matrix.shape[1] == 0 or b_matrix.shape[1] == 0:
        return np.zeros((a_matrix.shape[0], b_matrix.shape[0]))
    return cosine_similarity(a_matrix, b_matrix)


def char_similarity(name_a: str | None, name_b: str | None) -> float:
    """Longest-common-subsequence style ratio from difflib, in [0, 1]."""
    left, right = normalize(name_a), normalize(name_b)
    if not left or not right:
        return 0.0
    return difflib.SequenceMatcher(None, left, right).ratio()


def pair_features(
    a_row: pd.Series,
    b_row: pd.Series,
    name_sim: float,
    description_sim: float,
) -> dict[str, float]:
    """All features for one candidate pair.

    a_row and b_row are catalog rows; name_sim and description_sim come from the
    train-fitted TF-IDF objects in similarity.py.
    """
    tokens_a = set(tokenize(a_row["name"]))
    tokens_b = set(tokenize(b_row["name"]))
    models_a = model_tokens(a_row["name"])
    models_b = model_tokens(b_row["name"])
    numbers_a = numeric_tokens(a_row["name"])
    numbers_b = numeric_tokens(b_row["name"])

    price_a = parse_price(a_row["price"])
    price_b = parse_price(b_row["price"])
    both_prices_known = float(price_a is not None and price_b is not None)

    return {
        "name_similarity": float(name_sim),
        "description_similarity": float(description_sim),
        "token_overlap": _jaccard(tokens_a, tokens_b),
        "char_similarity": char_similarity(a_row["name"], b_row["name"]),
        "model_token_overlap": _jaccard(models_a, models_b),
        # 1.0 only when both sides name a specific model and they disagree.
        "model_token_conflict": float(bool(models_a) and bool(models_b) and not (models_a & models_b)),
        "numeric_token_overlap": _jaccard(numbers_a, numbers_b),
        "manufacturer_in_name": manufacturer_in_name(a_row["name"], b_row.get("manufacturer", "")),
        # 0.0 when either price is missing; read together with price_known_both.
        "price_difference": price_difference(price_a, price_b),
        "price_known_both": both_prices_known,
    }


def build_feature_frame(
    candidates: pd.DataFrame,
    abt: pd.DataFrame,
    buy: pd.DataFrame,
    similarity_model,
) -> pd.DataFrame:
    """Feature values for every row of a candidate list (idA, idB, rank).

    No labels here: use label_candidates to add them, and only for training pairs.
    """
    a_rows = abt.set_index("id").loc[candidates["idA"].tolist()].reset_index()
    b_rows = buy.set_index("id").loc[candidates["idB"].tolist()].reset_index()

    # cosine_similarity returns an n x m matrix, but the candidate list is an
    # aligned pairing, so each pair's value is on the diagonal.
    name_sim = np.diag(name_similarity(similarity_model, a_rows, b_rows))
    description_sim = np.diag(description_similarity(similarity_model, a_rows, b_rows))

    records = []
    for position, candidate in enumerate(candidates.itertuples(index=False)):
        row = pair_features(
            a_rows.iloc[position],
            b_rows.iloc[position],
            name_sim[position],
            description_sim[position],
        )
        row["idA"] = candidate.idA
        row["idB"] = candidate.idB
        row["rank"] = candidate.rank
        records.append(row)

    return pd.DataFrame.from_records(records)


def label_candidates(frame: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Attach label=1 for gold pairs, 0 for mined negatives."""
    gold_pairs = set(zip(gold["idAbt"], gold["idBuy"]))
    frame = frame.copy()
    frame["label"] = [
        int((id_a, id_b) in gold_pairs) for id_a, id_b in zip(frame["idA"], frame["idB"])
    ]
    return frame


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    """The numeric feature columns in the documented order."""
    return frame[FEATURE_NAMES].to_numpy(dtype=float)