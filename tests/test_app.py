"""Tests for the Streamlit demo.

AppTest runs the real script and drives its widgets, so a rendering-time error
fails here instead of in front of an interviewer. That is deliberate: a manual
browser test once found a crash that a helper-level test had missed.
"""

import ast
import importlib.util

import pytest
from joblib import load

from entity_resolution.blocking import retrieve
from entity_resolution.data import (
    PROJECT_ROOT,
    load_splits,
    records_in_split,
    split_records,
)
from entity_resolution.features import FEATURE_NAMES, build_feature_frame
from entity_resolution.model import DECISIONS, apply_decisions
from entity_resolution.features import fit_similarity

APP = PROJECT_ROOT / "app" / "streamlit_app.py"


def load_app_module():
    spec = importlib.util.spec_from_file_location("streamlit_app", APP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pipeline():
    bundle = load(PROJECT_ROOT / "models" / "matcher.joblib")
    abt, buy, gold = split_records()
    splits = load_splits()
    similarity_model = fit_similarity(
        records_in_split(abt, splits, "train"), records_in_split(buy, splits, "train")
    )
    return bundle, abt, buy, gold, similarity_model


def test_app_file_exists_and_parses():
    assert APP.exists()
    ast.parse(APP.read_text())


def test_app_imports_without_error():
    module = load_app_module()
    assert callable(module.main)
    assert module.DECISION_STYLE["MATCH"]
    assert module.DECISION_STYLE["REVIEW"]
    assert module.DECISION_STYLE["NO MATCH"]


def test_app_uses_the_frozen_artifacts(pipeline):
    bundle, *_ = pipeline
    assert (PROJECT_ROOT / "models" / "matcher.joblib").exists()
    assert bundle["model"].match_threshold == 0.9856394449


def test_app_uses_a_real_product_and_reaches_a_decision(pipeline):
    """A real Abt record must retrieve, score and decide."""
    bundle, abt, buy, _gold, similarity_model = pipeline
    model, retriever = bundle["model"], bundle["retriever"]

    record = abt[abt["name"].str.contains("Weber", case=False, na=False)].iloc[0]
    one = abt[abt["id"] == record["id"]].reset_index(drop=True)

    candidates = retrieve(retriever, one, buy, top_k=20)
    assert len(candidates) == 20

    features = build_feature_frame(candidates, abt, buy, similarity_model)
    decisions = apply_decisions(
        features, model.score(features), model.match_threshold, model.review_threshold
    )
    top = decisions.sort_values("score_rank").iloc[0]
    assert top["decision"] in DECISIONS
    assert 0.0 <= top["score"] <= 1.0


def test_demo_can_display_all_three_decisions(pipeline):
    """Each decision path must be reachable on real data, so the demo can show
    MATCH, REVIEW and NO MATCH."""
    bundle, abt, buy, _gold, similarity_model = pipeline
    model, retriever = bundle["model"], bundle["retriever"]

    candidates = retrieve(retriever, abt.reset_index(drop=True), buy, top_k=20)
    features = build_feature_frame(candidates, abt, buy, similarity_model)
    decisions = apply_decisions(
        features, model.score(features), model.match_threshold, model.review_threshold
    )
    top = decisions.sort_values(["idA", "score_rank"]).groupby("idA").first()
    seen = set(top["decision"])
    assert seen == {"MATCH", "REVIEW", "NO MATCH"}


def test_evidence_values_are_real_and_finite(pipeline):
    """Every feature shown as evidence must come from the pair, not a template."""
    bundle, abt, buy, _gold, similarity_model = pipeline
    model, retriever = bundle["model"], bundle["retriever"]

    record = abt.iloc[0]
    one = abt[abt["id"] == record["id"]].reset_index(drop=True)
    candidates = retrieve(retriever, one, buy, top_k=20)
    features = build_feature_frame(candidates, abt, buy, similarity_model)
    values = [float(features.iloc[0][name]) for name in FEATURE_NAMES]
    assert all(0.0 <= v <= 1.0 for v in values)
    assert any(v > 0 for v in values)


def test_every_feature_has_a_coefficient_to_explain_it(pipeline):
    bundle, *_ = pipeline
    coefficients = bundle["model"].coefficients()
    assert set(coefficients["feature"]) == set(FEATURE_NAMES)
    assert len(coefficients) == len(FEATURE_NAMES)


# ---- end-to-end run of the real app script through Streamlit's own harness ----
#
# The manual browser test caught a crash that the earlier headless smoke test
# missed, because that test exercised the helper functions but never ran the
# Streamlit script itself. AppTest runs the real script and drives the widgets,
# so a rendering-time error fails here.


def run_app():
    app_test = pytest.importorskip("streamlit.testing.v1")
    return app_test.AppTest.from_file(str(APP), default_timeout=180)


def test_app_script_runs_with_no_exceptions():
    app = run_app().run()
    assert not app.exception, [e.value for e in app.exception]


def test_app_renders_the_candidate_table_with_every_displayed_field():
    """The exact columns the crashed version could not select."""
    app = run_app().run()
    app.text_input[0].set_value("Weber").run()
    assert not app.exception, [e.value for e in app.exception]

    options = app.selectbox[0].options
    assert len(options) > 0
    app.selectbox[0].set_value(options[0]).run()
    assert not app.exception, [e.value for e in app.exception]

    table = app.dataframe[-1].value
    for column in (
        "Buy id",
        "Buy name",
        "manufacturer",
        "price",
        "retrieval similarity",
        "model score",
        "decision",
    ):
        assert column in table.columns, f"missing displayed column {column!r}"
    assert len(table) == 20
    assert not table["Buy name"].isna().any()


def test_app_renders_the_evidence_table():
    app = run_app().run()
    app.text_input[0].set_value("Weber").run()
    app.selectbox[0].set_value(app.selectbox[0].options[0]).run()

    evidence = app.dataframe[0].value
    assert list(evidence.columns) == ["feature", "value", "coefficient", "direction"]
    assert set(evidence["feature"]) == set(FEATURE_NAMES)


def test_app_shows_a_non_empty_price_for_at_least_one_candidate():
    """Blank prices were the easy case; prove real currency values render."""
    app = run_app().run()
    app.text_input[0].set_value("Sony").run()
    app.selectbox[0].set_value(app.selectbox[0].options[0]).run()

    table = app.dataframe[-1].value
    populated = table[table["price"].astype(str).str.startswith("$")]
    assert len(populated) > 0
    _abt, buy, _gold = split_records()
    row = populated.iloc[0]
    expected = buy.set_index("id").loc[row["Buy id"]]
    assert row["price"] == expected["price"]
    assert row["manufacturer"] == (expected["manufacturer"] or "")


def test_app_survives_a_search_with_no_matches():
    app = run_app().run()
    app.text_input[0].set_value("zzz_no_such_product_zzz").run()
    assert not app.exception, [e.value for e in app.exception]


def test_search_helper_filters_real_names():
    module = load_app_module()
    abt, _, _ = split_records()
    found = module.search_catalog(abt, "Weber", limit=10)
    assert len(found) > 0
    assert found["name"].str.contains("Weber", case=False).all()
    empty = module.search_catalog(abt, "zzz_no_such_product_zzz")
    assert empty.empty


# ---- regression: the manual browser test found this bug ----
#
# The app selected these columns straight off `decisions`:
#   ["score_rank","idB","name","manufacturer","price","cosine_sim","score","decision"]
# but `decisions` only ever carries feature values plus idA/idB/rank/score/
# score_rank/decision. Selecting them raised
#   KeyError: "['name', 'manufacturer', 'price', 'cosine_sim'] not in index"
# and crashed the demo for every product.


def test_decisions_frame_does_not_carry_catalog_columns(pipeline):
    """Document the shape that caused the bug, so the bug cannot be reintroduced."""
    bundle, abt, buy, _gold, similarity_model = pipeline
    model, retriever = bundle["model"], bundle["retriever"]

    one = abt.head(1).reset_index(drop=True)
    candidates = retrieve(retriever, one, buy, top_k=20)
    features = build_feature_frame(candidates, abt, buy, similarity_model)
    decisions = apply_decisions(
        features, model.score(features), model.match_threshold, model.review_threshold
    )

    for column in ("name", "manufacturer", "price", "cosine_sim"):
        assert column not in decisions.columns, (
            f"{column!r} must not be assumed present on decisions"
        )
    # The sources that do carry them.
    assert "cosine_sim" in candidates.columns
    assert {"name", "manufacturer", "price"} <= set(buy.columns)


def test_selecting_catalog_columns_off_decisions_still_raises_key_error(pipeline):
    """Reproduces the original crash, proving the test above is meaningful."""
    bundle, abt, buy, _gold, similarity_model = pipeline
    model, retriever = bundle["model"], bundle["retriever"]

    one = abt.head(1).reset_index(drop=True)
    candidates = retrieve(retriever, one, buy, top_k=20)
    features = build_feature_frame(candidates, abt, buy, similarity_model)
    decisions = apply_decisions(
        features, model.score(features), model.match_threshold, model.review_threshold
    )

    with pytest.raises(KeyError):
        decisions[
            ["score_rank", "idB", "name", "manufacturer", "price", "cosine_sim"]
        ]


def test_candidate_display_table_supplies_every_missing_column(pipeline):
    """The fix: join from the candidate list and the Buy catalog on idB."""
    module = load_app_module()
    bundle, abt, buy, _gold, similarity_model = pipeline
    model, retriever = bundle["model"], bundle["retriever"]

    record = abt[abt["name"].str.contains("Weber", case=False, na=False)].iloc[0]
    one = abt[abt["id"] == record["id"]].reset_index(drop=True)
    candidates = retrieve(retriever, one, buy, top_k=20)
    features = build_feature_frame(candidates, abt, buy, similarity_model)
    decisions = apply_decisions(
        features, model.score(features), model.match_threshold, model.review_threshold
    )

    table = module.candidate_display_table(decisions, candidates, buy)
    assert list(table.columns) == [
        "rank",
        "Buy id",
        "Buy name",
        "manufacturer",
        "price",
        "retrieval similarity",
        "model score",
        "decision",
    ]
    assert len(table) == 20


def test_candidate_display_table_values_come_from_the_real_catalog(pipeline):
    """Displayed name, manufacturer and price must match the Buy record exactly."""
    module = load_app_module()
    bundle, abt, buy, _gold, similarity_model = pipeline
    model, retriever = bundle["model"], bundle["retriever"]

    one = abt.head(1).reset_index(drop=True)
    candidates = retrieve(retriever, one, buy, top_k=20)
    features = build_feature_frame(candidates, abt, buy, similarity_model)
    decisions = apply_decisions(
        features, model.score(features), model.match_threshold, model.review_threshold
    )
    table = module.candidate_display_table(decisions, candidates, buy)
    lookup = buy.set_index("id")

    for row in table.itertuples(index=False):
        expected = lookup.loc[row[1]]
        assert row[2] == expected["name"]
        assert row[3] == (expected["manufacturer"] or "")
        assert row[4] == (expected["price"] or "")


def test_candidate_display_table_shows_the_real_retrieval_similarity(pipeline):
    module = load_app_module()
    bundle, abt, buy, _gold, similarity_model = pipeline
    model, retriever = bundle["model"], bundle["retriever"]

    one = abt.head(1).reset_index(drop=True)
    candidates = retrieve(retriever, one, buy, top_k=20)
    features = build_feature_frame(candidates, abt, buy, similarity_model)
    decisions = apply_decisions(
        features, model.score(features), model.match_threshold, model.review_threshold
    )
    table = module.candidate_display_table(decisions, candidates, buy)

    expected = dict(zip(candidates["idB"], candidates["cosine_sim"].round(4)))
    for row in table.itertuples(index=False):
        assert row[5] == expected[row[1]]
    assert table["retrieval similarity"].between(0.0, 1.0).all()


def test_candidate_display_table_is_ordered_by_model_score_rank(pipeline):
    module = load_app_module()
    bundle, abt, buy, _gold, similarity_model = pipeline
    model, retriever = bundle["model"], bundle["retriever"]

    one = abt.head(1).reset_index(drop=True)
    candidates = retrieve(retriever, one, buy, top_k=20)
    features = build_feature_frame(candidates, abt, buy, similarity_model)
    decisions = apply_decisions(
        features, model.score(features), model.match_threshold, model.review_threshold
    )
    table = module.candidate_display_table(decisions, candidates, buy)
    assert table["rank"].tolist() == sorted(table["rank"].tolist())
    assert table["model score"].is_monotonic_decreasing


def test_candidate_display_table_survives_missing_buy_fields(pipeline):
    """Buy rows can have empty manufacturer/price; display must not show NaN."""
    module = load_app_module()
    bundle, abt, buy, _gold, similarity_model = pipeline
    model, retriever = bundle["model"], bundle["retriever"]

    trimmed = buy.copy()
    trimmed.loc[trimmed.index[:5], ["manufacturer", "price"]] = ""
    one = abt.head(1).reset_index(drop=True)
    candidates = retrieve(retriever, one, trimmed, top_k=20)
    features = build_feature_frame(candidates, abt, trimmed, similarity_model)
    decisions = apply_decisions(
        features, model.score(features), model.match_threshold, model.review_threshold
    )
    table = module.candidate_display_table(decisions, candidates, trimmed)
    assert not table["manufacturer"].isna().any()
    assert not table["price"].isna().any()
    assert not table["Buy name"].isna().any()