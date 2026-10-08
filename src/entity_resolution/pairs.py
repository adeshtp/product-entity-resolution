"""Training pair mining.

Positives: TRAIN gold pairs whose B record is also in the TRAIN B pool. Since
the split keeps whole entity groups together, a TRAIN gold pair never points at a
validation B record, so this needs no extra filtering.

Negatives: mined from TRAIN A records against the **TRAIN B pool only**. This is
the important rule. If negatives were mined against all of Catalog B, the model
would be trained on pairs involving validation and test B records, which leaks
the shape of the evaluation set into the fitted weights.

Negative categories, in the order they are collected:
1. every non-gold TRAIN-A x TRAIN-B pair inside the retrieved candidate list
   (these are the hard ones: same manufacturer, similar wording, close price)
2. a bounded sample of easy negatives drawn from TRAIN A x TRAIN B pairs outside
   the candidate list, so the model also sees obvious non-matches
"""
import numpy as np
import pandas as pd

from entity_resolution.features import build_feature_frame, label_candidates

EASY_NEGATIVES_PER_A = 5
RANDOM_SEED = 42


def mine_training_pairs(
    train_candidates: pd.DataFrame,
    train_a: pd.DataFrame,
    train_b: pd.DataFrame,
    gold: pd.DataFrame,
    similarity_model,
    easy_negatives_per_a: int = EASY_NEGATIVES_PER_A,
) -> pd.DataFrame:
    """Build the labelled TRAIN pair set. Returns a labelled feature frame."""
    labelled = label_candidates(build_feature_frame(train_candidates, train_a, train_b, similarity_model), gold)

    hard_negatives = labelled[labelled["label"] == 0].copy()
    positives = labelled[labelled["label"] == 1].copy()

    easy = _sample_easy_negatives(train_candidates, train_a, train_b, easy_negatives_per_a)
    easy_frame = label_candidates(
        build_feature_frame(easy, train_a, train_b, similarity_model), gold
    )
    easy_frame = easy_frame[easy_frame["label"] == 0].copy()

    combined = pd.concat([positives, hard_negatives, easy_frame], ignore_index=True)
    return combined.sort_values(["idA", "rank"]).reset_index(drop=True)


def _sample_easy_negatives(
    train_candidates: pd.DataFrame,
    train_a: pd.DataFrame,
    train_b: pd.DataFrame,
    per_a: int,
) -> pd.DataFrame:
    """Sample non-gold TRAIN-A x TRAIN-B pairs that retrieval did not surface.

    These are the easy negatives: nothing lexically connects them to the A
    record, so without them the model would only ever see hard cases.
    """
    already = set(zip(train_candidates["idA"], train_candidates["idB"]))
    b_ids = train_b["id"].tolist()
    rng = np.random.default_rng(RANDOM_SEED)

    records = []
    for id_a in train_a["id"].tolist():
        pool = [id_b for id_b in b_ids if (id_a, id_b) not in already]
        if not pool:
            continue
        chosen = rng.choice(pool, size=min(per_a, len(pool)), replace=False)
        for rank, id_b in enumerate(chosen, start=1):
            records.append({"idA": id_a, "rank": rank, "idB": id_b})
    return pd.DataFrame.from_records(records, columns=["idA", "rank", "idB"])