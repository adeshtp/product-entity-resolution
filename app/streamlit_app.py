"""Local Streamlit demo for the product entity-resolution matcher.

Run:  streamlit run app/streamlit_app.py

Reads the frozen artefacts in models/matcher.joblib. Shows one real Abt product,
its retrieved Buy candidates, the real feature values behind each score, the
model score, and the MATCH / REVIEW / NO MATCH decision.

Every number on screen is computed from real data. Nothing is invented and no
text is templated to look like evidence.

No API keys, no external services, no network access.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from entity_resolution.blocking import retrieve  # noqa: E402
from entity_resolution.data import PROJECT_ROOT as DATA_ROOT  # noqa: E402
from entity_resolution.features import FEATURE_NAMES, build_feature_frame  # noqa: E402
from entity_resolution.model import apply_decisions  # noqa: E402
from entity_resolution.data import records_in_split, split_records  # noqa: E402
from entity_resolution.features import fit_similarity  # noqa: E402

DECISION_STYLE = {
    "MATCH": "✅",
    "REVIEW": "⚠️",
    "NO MATCH": "❌",
}


def candidate_display_table(
    decisions: pd.DataFrame, candidates: pd.DataFrame, buy: pd.DataFrame
) -> pd.DataFrame:
    """Join candidate metadata onto the decisions for display.

    Why this join exists: `build_feature_frame` deliberately returns only the
    feature values plus idA / idB / rank, so `decisions` never carried the Buy
    product fields, and `retrieve` keeps `cosine_sim` in the candidate list rather
    than in the features. Selecting those columns straight off `decisions`
    raised KeyError. Both sources already exist, so this joins them on idB rather
    than copying catalog data into the feature frame.
    """
    table = decisions.sort_values("score_rank").copy()

    catalog = buy[["id", "name", "manufacturer", "price"]].rename(
        columns={"id": "idB", "name": "Buy name"}
    )
    table = table.merge(catalog, on="idB", how="left")

    similarity = candidates[["idB", "cosine_sim"]].rename(
        columns={"cosine_sim": "retrieval similarity"}
    )
    table = table.merge(similarity, on="idB", how="left")

    table["Buy name"] = table["Buy name"].fillna("(not found)")
    table["manufacturer"] = table["manufacturer"].fillna("")
    table["price"] = table["price"].fillna("")
    table["retrieval similarity"] = table["retrieval similarity"].round(4)
    table["score"] = table["score"].round(4)

    return table[
        [
            "score_rank",
            "idB",
            "Buy name",
            "manufacturer",
            "price",
            "retrieval similarity",
            "score",
            "decision",
        ]
    ].rename(
        columns={
            "score_rank": "rank",
            "idB": "Buy id",
            "score": "model score",
        }
    )


@st.cache_resource
def load_artifacts():
    """Load the frozen matcher and refit the train-only transforms."""
    from joblib import load

    bundle = load(DATA_ROOT / "models" / "matcher.joblib")
    abt, buy, gold = split_records()
    train_a = records_in_split(abt, load_splits_cached(), "train")
    train_b = records_in_split(buy, load_splits_cached(), "train")
    similarity_model = fit_similarity(train_a, train_b)
    return bundle, abt, buy, similarity_model


@st.cache_resource
def load_splits_cached():
    from entity_resolution.data import load_splits

    return load_splits()


@st.cache_data
def search_catalog(abt: pd.DataFrame, query: str, limit: int = 40) -> pd.DataFrame:
    """Simple substring search over Abt product names."""
    if not query:
        return abt.head(limit)
    mask = abt["name"].str.contains(query, case=False, regex=False, na=False)
    return abt[mask].head(limit)


def _text(value) -> str:
    """Show blank catalog fields as an em dash instead of an empty gap."""
    text = str(value).strip()
    return text if text else "—"


DECISION_MESSAGE = {
    "MATCH": ("Auto-merged. This pair cleared the MATCH threshold, so no human check was needed.", "success"),
    "REVIEW": ("Sent to the review queue: above the REVIEW line but not confident enough to merge.", "warning"),
    "NO MATCH": ("No match. Nothing scored high enough to merge or to review.", "error"),
}


def main() -> None:
    st.set_page_config(page_title="Product Entity Resolution", layout="wide")
    st.title("Product Entity Resolution")
    st.caption(
        "Matching an Abt product against the Buy catalog. Retrieval is char n-gram "
        "TF-IDF fitted on train data only; the matcher is Logistic Regression over "
        "10 interpretable features."
    )

    bundle, abt, buy, similarity_model = load_artifacts()
    model = bundle["model"]
    retriever = bundle["retriever"]

    st.sidebar.header("Frozen decision policy")
    st.sidebar.metric("MATCH threshold", f"{model.match_threshold:.6f}")
    st.sidebar.metric("REVIEW threshold", f"{model.review_threshold:.4f}")
    st.sidebar.markdown(
        "At most **one** MATCH per Abt product, and a product is never forced to match."
    )
    st.sidebar.caption(
        "Chosen on the validation split and frozen before the test split was read."
    )
    st.sidebar.divider()
    st.sidebar.caption(
        "Every Abt product in this benchmark has a true Buy match, so the demo can "
        "show NO MATCH but the benchmark cannot measure real NO MATCH performance."
    )

    st.subheader("1. Choose an Abt product")
    query = st.text_input(
        "Search Abt product names", value="", placeholder="e.g. Sony, Weber, microwave"
    )
    matching = search_catalog(abt, query)

    if matching.empty:
        st.warning("No Abt product matched that search. Try a shorter term.")
        return

    options = {f"{row['id']}  ·  {row['name']}": row["id"] for _, row in matching.iterrows()}
    id_a = options[st.selectbox("Abt product", list(options))]
    a_record = abt.set_index("id").loc[id_a]

    st.subheader("2. The Abt record")
    name_col, detail_col, price_col = st.columns([3, 5, 2])
    with name_col:
        st.markdown("**Name**")
        st.write(_text(a_record["name"]))
    with detail_col:
        st.markdown("**Description**")
        st.write(_text(a_record["description"]))
        st.caption("Catalog A has no manufacturer column in the official files.")
    with price_col:
        st.markdown("**Price**")
        st.write(_text(a_record["price"]))
        st.caption(f"id {id_a}")

    one_a = abt[abt["id"] == id_a].reset_index(drop=True)
    candidates = retrieve(retriever, one_a, buy, top_k=20)
    features = build_feature_frame(candidates, abt, buy, similarity_model)
    decisions = apply_decisions(
        features, model.score(features), model.match_threshold, model.review_threshold
    )
    top = decisions.sort_values("score_rank").iloc[0]
    decision = top["decision"]

    st.subheader("3. Decision")
    score_col, banner_col = st.columns([2, 5])
    with score_col:
        st.metric(f"{DECISION_STYLE[decision]}  {decision}", f"{float(top['score']):.4f}")
        st.caption("model score (estimated probability)")
    with banner_col:
        message, kind = DECISION_MESSAGE[decision]
        getattr(st, kind)(message)

    st.subheader("4. Why — the feature values behind that score")
    coefficients = dict(
        zip(model.coefficients()["feature"], model.coefficients()["coefficient"])
    )
    evidence = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "value": [float(top[name]) for name in FEATURE_NAMES],
            "coefficient": [coefficients[name] for name in FEATURE_NAMES],
        }
    )
    evidence["direction"] = evidence["coefficient"].apply(
        lambda c: "toward match" if c > 0 else "away from match"
    )
    st.dataframe(
        evidence.round({"value": 4, "coefficient": 4}),
        width="stretch",
        hide_index=True,
    )

    st.subheader("5. Retrieved Buy candidates")
    st.dataframe(
        candidate_display_table(decisions, candidates, buy),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Retrieval cosine similarity ranks candidates on wording alone. The model "
        "score also uses price, manufacturer and model-number evidence, which is "
        "why the two orderings can differ."
    )

    st.subheader("6. The matched (or top) Buy record")
    b_record = buy.set_index("id").loc[top["idB"]]
    name_col, detail_col, price_col = st.columns([3, 5, 2])
    with name_col:
        st.markdown("**Name**")
        st.write(_text(b_record["name"]))
    with detail_col:
        st.markdown("**Description**")
        st.write(_text(b_record["description"]))
    with price_col:
        st.markdown("**Manufacturer / price**")
        st.write(_text(b_record.get("manufacturer", "")))
        st.write(_text(b_record["price"]))


if __name__ == "__main__":
    main()