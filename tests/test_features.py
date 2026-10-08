"""Tests for features.py. Run: pytest"""

import numpy as np
import pandas as pd
import pytest

from entity_resolution.features import (
    FEATURE_NAMES,
    build_feature_frame,
    char_similarity,
    feature_matrix,
    label_candidates,
    pair_features,
)
from entity_resolution.features import fit_similarity


def catalog(names, prices=None, manufacturers=None, ids=None, descriptions=None):
    count = len(names)
    return pd.DataFrame(
        {
            "id": ids or [str(i) for i in range(count)],
            "name": names,
            "description": descriptions if descriptions is not None else ["desc " + n for n in names],
            "price": prices if prices is not None else [""] * count,
            **({"manufacturer": manufacturers} if manufacturers is not None else {}),
        }
    )


def a_row(**kwargs):
    base = {"name": "Sony Turntable PSLX350H", "description": "belt drive", "price": "$399.00"}
    base.update(kwargs)
    return pd.Series(base)


def b_row(**kwargs):
    base = {
        "name": "Sony PSLX350H Turntable",
        "description": "belt drive turntable",
        "manufacturer": "Sony",
        "price": "$399.00",
    }
    base.update(kwargs)
    return pd.Series(base)


def test_feature_names_are_unique():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))


def test_pair_features_returns_every_documented_feature():
    features = pair_features(a_row(), b_row(), 0.9, 0.5)
    assert set(features) == set(FEATURE_NAMES)


def test_identical_product_scores_highly():
    features = pair_features(a_row(), b_row(), 0.9, 0.8)
    assert features["token_overlap"] > 0.5
    assert features["model_token_overlap"] == 1.0
    assert features["model_token_conflict"] == 0.0
    assert features["price_difference"] == 0.0
    assert features["price_known_both"] == 1.0


def test_different_model_number_is_flagged_as_a_conflict():
    """15 inch vs 17 inch monitors: nearly identical except the model.

    Model tokens need 3+ digits or mixed letters+digits, so the sizes are
    written the way a manufacturer writes them: W154 and W174.
    """
    features = pair_features(
        a_row(name="Sony W154 15 inch Monitor", price=""),
        b_row(name="Sony W174 17 inch Monitor", manufacturer="Sony", price=""),
        name_sim=0.8,
        description_sim=0.1,
    )
    assert features["model_token_conflict"] == 1.0
    assert features["model_token_overlap"] == 0.0


def test_short_numbers_are_not_model_tokens():
    # "15 inch" vs "17 inch" must not be treated as a model-number conflict:
    # every product has some small number in its name, so this feature would
    # fire on almost every pair and carry no information.
    features = pair_features(
        a_row(name="Sony 15 inch Monitor", price=""),
        b_row(name="Sony 17 inch Monitor", manufacturer="Sony", price=""),
        name_sim=0.8,
        description_sim=0.1,
    )
    assert features["model_token_conflict"] == 0.0


def test_missing_price_is_distinguishable_from_equal_prices():
    missing = pair_features(a_row(price=""), b_row(price="$399.00"), 0.5, 0.5)
    equal = pair_features(a_row(price="$399.00"), b_row(price="$399.00"), 0.5, 0.5)
    # Both give price_difference 0.0 ...
    assert missing["price_difference"] == equal["price_difference"] == 0.0
    # ... but the indicator is what separates them. Without it a missing price
    # would be indistinguishable from two equal prices.
    assert missing["price_known_both"] == 0.0
    assert equal["price_known_both"] == 1.0


def test_price_known_both_is_zero_when_either_side_missing():
    for a_price, b_price in (("", "$10"), ("$10", ""), ("", "")):
        features = pair_features(a_row(price=a_price), b_row(price=b_price), 0.1, 0.1)
        assert features["price_known_both"] == 0.0


def test_manufacturer_in_name_is_one_directional():
    """A has no manufacturer column, so this can only check B's brand in A's name."""
    present = pair_features(a_row(name="Sony Turntable"), b_row(manufacturer="Sony"), 0.5, 0.5)
    absent = pair_features(a_row(name="Sony Turntable"), b_row(manufacturer="Bose"), 0.5, 0.5)
    assert present["manufacturer_in_name"] == 1.0
    assert absent["manufacturer_in_name"] == 0.0
    assert absent["manufacturer_in_name"] == 0.0  # never evidence against


def test_manufacturer_feature_handles_empty_brand():
    features = pair_features(a_row(), b_row(manufacturer=""), 0.5, 0.5)
    assert features["manufacturer_in_name"] == 0.0


def test_features_survive_empty_text_everywhere():
    features = pair_features(
        a_row(name="", description="", price=""),
        b_row(name="", description="", manufacturer="", price=""),
        name_sim=0.0,
        description_sim=0.0,
    )
    values = feature_matrix(pd.DataFrame([features]))
    assert np.isfinite(values).all()


def test_char_similarity_bounds():
    assert char_similarity("abc", "abc") == 1.0
    assert char_similarity("", "abc") == 0.0
    assert 0.0 <= char_similarity("Sony Turntable", "Bose Speaker") < 1.0


def test_label_candidates_marks_gold_pairs():
    frame = pd.DataFrame({"idA": ["1", "1"], "idB": ["10", "11"]})
    gold = pd.DataFrame({"idAbt": ["1"], "idBuy": ["10"]})
    labelled = label_candidates(frame, gold)
    assert labelled["label"].tolist() == [1, 0]


def test_build_feature_frame_end_to_end():
    abt = catalog(["Sony Turntable PSLX350H"], ["$399.00"], ids=["1"])
    buy = catalog(
        ["Sony PSLX350H Turntable", "Bose Acoustimass Speaker"],
        ["$399.00", "$200.00"],
        manufacturers=["Sony", "Bose"],
        ids=["10", "11"],
    )
    similarity_model = fit_similarity(abt, buy)
    candidates = pd.DataFrame({"idA": ["1", "1"], "rank": [1, 2], "idB": ["10", "11"]})
    frame = build_feature_frame(candidates, abt, buy, similarity_model)
    assert len(frame) == 2
    assert frame.loc[0, "manufacturer_in_name"] == 1.0
    assert frame.loc[1, "manufacturer_in_name"] == 0.0
    assert frame.loc[0, "name_similarity"] > frame.loc[1, "name_similarity"]
    assert np.isfinite(feature_matrix(frame)).all()


def test_build_feature_frame_handles_empty_descriptions():
    abt = catalog(["Sony PSLX350H"], ["$399.00"], ids=["1"])
    buy = catalog(["Sony PSLX350H"], ["$399.00"], manufacturers=["Sony"], ids=["10"], descriptions=[""])
    similarity_model = fit_similarity(abt, buy)
    candidates = pd.DataFrame({"idA": ["1"], "rank": [1], "idB": ["10"]})
    frame = build_feature_frame(candidates, abt, buy, similarity_model)
    assert frame.loc[0, "description_similarity"] == 0.0
    assert np.isfinite(feature_matrix(frame)).all()


def test_feature_matrix_column_order_matches_feature_names():
    abt = catalog(["A name"], ["$1.00"], ids=["1"])
    buy = catalog(["A name"], ["$1.00"], manufacturers=["A"], ids=["10"])
    similarity_model = fit_similarity(abt, buy)
    candidates = pd.DataFrame({"idA": ["1"], "rank": [1], "idB": ["10"]})
    frame = build_feature_frame(candidates, abt, buy, similarity_model)
    assert list(frame[FEATURE_NAMES].columns) == FEATURE_NAMES