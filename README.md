# Product Entity Resolution

Given a product in catalog A, which product in catalog B is the same real-world product?

Built on the official [Abt-Buy benchmark](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution)
(University of Leipzig Database Group): 1,081 Abt products, 1,092 Buy products,
1,097 gold matches. Everything runs locally on CPU with no API keys or paid
services.

## Result (held-out test split)

| Metric | Test |
| --- | --- |
| Pair precision | 0.8571 |
| Pair recall | 0.5488 |
| F1 | 0.6691 |
| False merges | 15 |
| Hits@1 / 5 / 10 (ranked by model) | 0.8110 / 0.9817 / 0.9939 |
| MRR | 0.8810 |
| Candidate Recall@5 / 10 / 20 | 0.9573 / 0.9939 / 1.0000 |
| Reduction ratio (pairs removed by blocking) | 98.17% |
| Review queue rate | 0.1625 |

The system targets 0.95 precision and **does not reach it on test**. The
threshold was chosen on validation to hit exactly 0.95, which left it with no
margin; on unseen data it delivers 0.8571 (bootstrap 95% CI 0.7857–0.9214).
That gap is reported rather than tuned away, because the test split was read once
after everything else was frozen.

## Why this is not a lookup

Only 10 of the 1,097 gold pairs have identical names after normalization, so
exact matching recovers about 1% of the mapping. The same product is written very
differently by the two shops:

```
Abt : Sony Turntable - PSLX350H / Belt Drive System / 33-1/3 and 45 RPM Speeds
Buy : Sony PSLX350H Turntable, Belt Drive, Silver
```

The task is to decide which pairs are the same product **and which are not**,
because a wrong merge silently corrupts a catalog.

## Approach

```
normalize -> retrieve candidates -> features -> Logistic Regression -> decision
```

**Retrieval** is char n-gram TF-IDF over normalized names, cosine similarity,
top-20, against the whole Buy catalog. Character n-grams survive the spelling
drift between catalogs where word tokens do not. It removes 98.17% of the
possible pairs and loses no gold match at k=20.

**Features** are ten interpretable values per pair: name and description TF-IDF
cosine, token overlap, character similarity, model-token overlap and conflict,
numeric-token overlap, manufacturer-in-name, and price difference with an explicit
"both prices known" flag. Numbers are never stripped, because model numbers and
capacities carry the product's identity.

**The matcher** is Logistic Regression over those ten features, with
`class_weight="balanced"` on 766 positive and 18,134 negative training pairs. The
negatives are mined from train A against the train B pool only, so validation and
test records never enter training.

**The decision** is MATCH / REVIEW / NO MATCH, with at most one MATCH per Abt
product. Only the top-scoring candidate may be merged, and a product is never
forced to match. Anything uncertain goes to a human queue.

## Data

Encoding is `cp1252`, not UTF-8 (`Abt.csv` fails UTF-8 decoding). Catalog A has no
manufacturer column even though the benchmark page lists one, so manufacturer
agreement cannot be a symmetric comparison; the substitute feature checks whether
the Buy brand appears in the Abt name, which is one-directional and only ever
evidence *for* a match.

The mapping is not one-to-one: as a graph it has 1,076 connected components, 16 of
them 1A+2B. That is why the split is built on components rather than rows.

## Split and leakage

Split by connected component of the gold graph, 70/15/15, seed 42. Verified: 0 of
1,097 gold pairs cross a boundary.

Everything learned is fitted on train only — the retrieval TF-IDF, both feature
TF-IDF objects, and the scaler, which sits inside the Pipeline so it sees only
training rows. Validation was used for model and threshold decisions. The test
split was read once, at the end.

## Honest limitations

- **Real-world NO MATCH performance cannot be measured here.** Every Abt product
  in the benchmark has a gold match, so there is no genuinely unmatched record.
  No such claim is made. A diagnostic probe that removes a product's gold partner
  and checks whether the system still invents a match is reported separately and
  labelled as a probe, not a measurement.
- Recall is low by design: 0.5488 of gold pairs become automatic merges, but
  0.9451 are merged or queued for a human. Both numbers are reported.
- The dominant error is a same-brand lookalike with a different model number. The
  fitted coefficients show why: `name_similarity` carries +3.19 while
  `model_token_conflict` carries only −0.74, so lexical similarity outvotes the
  evidence that two products differ.
- 164 test and 161 validation products are small samples; bootstrap intervals are
  correspondingly wide.

## Layout

```
src/entity_resolution/   data, normalize, blocking, similarity, features,
                         pairs, model, evaluation, train, evaluate_test
app/streamlit_app.py     local demo
reports/                 metrics.json, test_metrics.json, error_analysis.csv
docs/                    methodology.md, learning_notes.md
tests/                   169 tests
```

## Running it

```bash
python -m venv .venv --system-site-packages
.venv/bin/pip install -r requirements.txt

# data is not committed; see data/README.md for the download
mkdir -p data/raw
curl -L -o /tmp/Abt-Buy.zip https://dbs.uni-leipzig.de/files/datasets/Abt-Buy.zip
unzip -o /tmp/Abt-Buy.zip -d data/raw

PYTHONPATH=src .venv/bin/python -m entity_resolution.data        # write the split
PYTHONPATH=src .venv/bin/python -m entity_resolution.train        # train + validation
PYTHONPATH=src .venv/bin/python -m entity_resolution.evaluate_test  # test split, once
PYTHONPATH=src .venv/bin/python -m pytest
PYTHONPATH=src .venv/bin/streamlit run app/streamlit_app.py
```

`PYTHONPATH=src` is needed because Python 3.14 ignores editable-install `.pth`
files.

## Tests

169 tests covering normalization, retrieval, features, the matcher and decision
policy, evaluation definitions, and the Streamlit app. The app tests run the real
script through Streamlit's `AppTest`, so a rendering error fails in the suite
rather than in front of you.

## Notes

- `reports/*.json` are the source of truth for every number quoted here.
- This project grew from earlier NLP coursework where I had used TF-IDF and
  Logistic Regression on review text. Reusing that pipeline, I found and fixed
  here a mistake present in those notebooks: the TF-IDF was fitted on all the
  data before the train/test split, which makes the test score optimistic.
- Tested on Python 3.14.3 with pandas 3.0.0, scikit-learn 1.8.0, joblib, pytest
  and Streamlit; see `requirements.txt`.