# Methodology

How the system works and why each choice was made. Numbers here come from
`reports/metrics.json` (validation) and `reports/test_metrics.json` (test).

## 1. Data

The Abt-Buy benchmark from the University of Leipzig Database Group: 1,081 Abt
products, 1,092 Buy products, 1,097 gold pairs. Files were verified byte-for-byte
against the official download; checksums and provenance are in `data/README.md`.

Two properties of the data shaped the design:

**The files are cp1252, not UTF-8.** `Abt.csv` contains `®`, `°` and a soft
hyphen and raises `UnicodeDecodeError` under UTF-8, so everything is read with
`encoding="cp1252"`.

**Catalog A has no manufacturer column.** The benchmark page lists manufacturer
as a task attribute, but `Abt.csv` has only id, name, description, price.
Manufacturer agreement therefore cannot be a symmetric comparison between the two
sides.

Measured data quality:

| | Abt | Buy |
| --- | --- | --- |
| Rows | 1,081 | 1,092 |
| Empty price | 663 | 502 |
| Empty description | 0 | 441 |
| Duplicate names | 0 | 13 |

Prices are strings like `$1,000.00` and must be cleaned before numeric use. Every
present value parses once `$` and thousands separators are stripped; none is
discarded.

### Mapping structure

The gold mapping is not one-to-one. Treating it as a bipartite graph and taking
connected components gives **1,076 entity groups**: 1,055 are 1A+1B, 16 are
1A+2B, and 5 are 2A+1B. Every A record and every B record appears in the mapping,
so there are no unmatched records.

Only 10 of the 1,097 gold pairs have identical names after normalization.

## 2. Normalization

`normalize.py` lowercases, folds unicode to ASCII, and collapses punctuation, with
two deliberate choices:

**Numbers are never stripped.** Model numbers (`PSLX350H`, `EZXS88W`) and
capacities (`2.0 cu.ft.`) identify a specific variant. Removing digits would merge
a 15-inch monitor with a 17-inch one.

**Alphanumeric runs stay whole.** `pslx350h` is one token, not three. Splitting
it discards the model number, which is the most identity-bearing token in the
data. A negative lookahead in the token pattern stops the number branch from
claiming the digits out of a mixed token, so `10ft` survives as one token too.

Unmappable symbols become a space rather than being deleted, because deleting them
fuses the words on either side (`DVD±R` would become `dvdr`).

## 3. Split

The assignment unit is the **connected component** of the gold graph, so a whole
product relationship moves together. Components are shuffled with seed 42 and
sliced 70/15/15 by group count.

| Split | Groups | A records | Gold pairs |
| --- | --- | --- | --- |
| train | 753 | 756 | 767 |
| val | 161 | 161 | 166 |
| test | 162 | 164 | 164 |

A row-level split would place one side of a 2A+1B group in train and the other in
test, so the matcher would be scored on a twin it had already seen. Verified
programmatically: 1,076 unique groups, no overlap between splits, full coverage,
and **0 of 1,097 gold pairs cross a boundary**.

The split files are gitignored, so `python -m entity_resolution.data` regenerates
them; the same command reproduces the same files byte-for-byte.

## 4. Candidate retrieval

Char n-gram TF-IDF (`char_wb`, 3–4 grams) over normalized names, cosine
similarity, top-20, against all 1,092 Buy records.

Character n-grams because the same product is spelled differently across the two
catalogs: `Sony Turntable PSLX350H` and `Sony PSLX 350H Turntable Belt Drive`
share few whole words but many character fragments.

Retrieval is fitted on train A and train B names only; validation and test names
are transformed with the frozen vectorizer.

The candidate pool is the whole Buy catalog, because the real task is matching a
newly arrived product against an existing one. A pool restricted to the same
split's B records would measure the matcher in isolation but would not resemble
the deployment question, so it is not used for any reported number.

| Test | Value |
| --- | --- |
| Candidate Recall@5 / 10 / 20 | 0.9573 / 0.9939 / 1.0000 |
| Gold matches lost at k=20 | 0 |
| Remaining candidate fraction | 0.018315 |
| Reduction ratio | 0.981685 |

Blocking removes 98.17% of the possible pairs and loses no gold match, so
retrieval is not the bottleneck. The matcher is.

## 5. Features

Ten values per candidate pair:

| Feature | Why |
| --- | --- |
| `name_similarity` | main lexical signal, char n-gram TF-IDF cosine |
| `description_similarity` | free-form marketing text, word TF-IDF cosine |
| `token_overlap` | Jaccard on word tokens; reads differently from the cosine |
| `char_similarity` | sequence ratio, robust to reordering |
| `model_token_overlap` | Jaccard on model tokens; separates a 15-inch from a 17-inch monitor |
| `model_token_conflict` | both sides name a model and they disagree |
| `numeric_token_overlap` | agreement on capacities and sizes |
| `manufacturer_in_name` | Buy brand appears in the Abt name |
| `price_difference` | relative gap, defined as 0 when either side is missing |
| `price_known_both` | explicit flag so a missing price is not read as an equal price |

A model token is an alphanumeric run mixing letters and digits (`PSLX350H`,
`EZXS88W`, `10ft`) or a pure run of three or more digits. Short numbers are
excluded on purpose: every product name contains some small number, so treating
them as model tokens would make the conflict feature fire on nearly every pair.

`price_difference` returns 0 when either price is missing, which on its own would
make a missing price indistinguishable from two identical prices. The separate
`price_known_both` flag is what carries that distinction.

Retrieval rank is deliberately not a feature. It already encodes the same lexical
evidence as `name_similarity` and would blur the retrieval/matching boundary.

### The manufacturer substitute

Since A has no manufacturer, agreement cannot be symmetric. The substitute asks
whether the Buy manufacturer string appears in the normalized Abt name. It is
deterministic and needs no training, but its limits are real: it cannot detect a
correct pair whose Abt name omits the brand; a short brand such as `GE` matches by
accident; and the 116 manufacturer values are not normalized against each other,
so a 0 usually means unknown rather than contradiction. It only ever provides
evidence **for** a match, never against one.

## 6. Training pairs

Positives are the train gold pairs. Negatives are mined from train A against the
train B pool only — 14,354 hard negatives from the retrieved candidate list, plus
3,780 easy negatives sampled from outside it. Mining against the whole catalog
would put validation and test B records into training and flatter the evaluation,
so `train.py` raises at runtime if any training pair points outside the train B
pool, and a test asserts the same thing.

766 positives against 18,134 negatives, a positive rate of 0.0405. The imbalance
is handled with `class_weight="balanced"` rather than resampling, because
resampling would change which pairs exist and discard part of the mined hard
negative structure.

## 7. Model and thresholds

Logistic Regression in a `Pipeline` with `StandardScaler` and
`class_weight="balanced"`, seed 42. Outputs are reported as model score or
estimated probability, never as confidence, because calibration was not
evaluated.

The MATCH threshold is chosen by a rule registered before validation was read:
*the lowest achievable threshold at which the MATCH decisions reach 0.95
precision on validation*, measured by running the full decision policy so
precision matches what actually gets reported. The REVIEW threshold is the 95th
percentile of train negative scores, so the review line never sees validation.

Candidate thresholds are the sorted unique validation scores. A decision uses
`score >= threshold`, so the MATCH set only changes when the threshold crosses an
observed score; every other value reproduces one of these. An earlier version
swept a fixed 0.05 grid capped at 0.95, which made the rule unable to satisfy its
own target, and that cap was removed.

Frozen values: **MATCH 0.9856394449**, **REVIEW 0.5182070609**.

## 8. Decision policy

Candidates for a product are ranked by model score. Only the top-ranked candidate
may become MATCH, and only above the MATCH threshold, so one Abt product can never
be merged twice. A candidate between the two thresholds becomes REVIEW,
everything else is NO MATCH, and a product is never forced to match.

On test: 105 MATCH / 533 REVIEW / 2,642 NO MATCH, with 105 of 164 products merged.

## 9. Evaluation

Retrieval and matching are measured separately and never mixed, because a gold
pair lost during retrieval cannot be recovered by a better matcher.

| Metric | Question |
| --- | --- |
| Candidate Recall@K | can the right Buy record be found at all? |
| Pair precision / recall / F1 | of the pairs merged, how many were real? |
| Product Hits@K / MRR | did the matcher rank the right record first? |
| MATCH-only recall | how many gold pairs became an automatic merge? |
| MATCH+REVIEW coverage | ... including those queued for a human? |
| Review rate | share of candidate pairs sent to a human |

Hits@K is computed twice, once ranking by retrieval order and once by model score.
Ranking by retrieval reproduces candidate Recall@K by construction and says
nothing about the matcher; the model ranking is the real product-level measure.
Reporting only the retrieval version once made the two look identical, which was
misleading.

Pair-level recall is capped at 1,081/1,097 = 0.9854, because 16 products have two
gold partners and may receive only one B record. Candidate recall has no such cap,
since a hit means any gold partner in the top K.

### Test results

| Measure | Test |
| --- | --- |
| Precision / recall / F1 | 0.8571 / 0.5488 / 0.6691 |
| Macro-F1 | 0.8275 |
| False merges / false non-matches | 15 / 74 |
| Hits@1 / 5 / 10 (model) | 0.8110 / 0.9817 / 0.9939 |
| MRR (model) | 0.8810 |
| MATCH-only recall | 0.5488 |
| MATCH+REVIEW coverage | 0.9451 |
| Review rate | 0.1625 |

Confusion matrix `[[3101, 15], [74, 90]]`. Of the 74 false non-matches, 65 went to
REVIEW and only 9 were rejected outright, so the strict threshold routes gold
pairs to a human queue rather than discarding them.

A percentile bootstrap resampled **by Abt product** (1,000 resamples, 164 units)
gives precision [0.7857, 0.9214], recall [0.4695, 0.6280], F1 [0.5946, 0.7394] and
Hits@1 [0.7134, 0.8415]. Resampling pairs instead of products would treat one
product's twenty candidates and its single outcome as independent and understate
the interval.

## 10. Error analysis

`reports/error_analysis.csv` holds one row per product that went wrong, for both
splits, with the real feature values behind each decision. Categories are precise:

| Category | Meaning |
| --- | --- |
| `false_merge` | a MATCH was emitted on a pair that is not gold |
| `gold_not_ranked_first` | a gold partner was retrieved but the matcher preferred another candidate — a matching error |
| `correct_but_sent_to_review` | the top candidate was gold but was queued for a human; not a false negative, since the system never claimed a non-match |
| `rejected_despite_being_top` | the top candidate was gold and was rejected outright |
| `gold_never_retrieved` | no gold partner was in the candidate list at all |

There are zero `gold_never_retrieved` rows, because candidate Recall@20 is 1.0000.
An earlier version labelled mis-ranked candidates as retrieval failures, which
blamed retrieval for a matching error; the categories are now derived from the
data.

The dominant error is a same-brand lookalike with a different model number —
several distinct Panasonic DECT phones collapsing onto one model, or a 52-inch
TV matched to a 42-inch one. The coefficients show the mechanism: `name_similarity`
carries +3.19 while `model_token_conflict` carries only −0.74, so lexical
similarity outvotes the evidence that the products differ. Mining extra same-brand
negatives was tried and rejected, because it increased false merges rather than
reducing them.

## 11. What this benchmark cannot measure

Every Abt product has at least one gold match, so there is no genuinely unmatched
record and real-world NO MATCH performance cannot be measured. No such claim is
made anywhere in this project.

The abstention probe removes a product's own gold partner from its candidate pool
and checks whether the system still emits a MATCH. On test, 30 of 164 products
still had a candidate above the MATCH boundary. It is a diagnostic, not a
benchmark precision figure, and it was used to select nothing.

## 12. Reproducibility

Fixed seeds throughout. The split regenerates byte-for-byte. `evaluate_test.py`
is deliberately separate from `train.py` so the training pipeline is untouched by
the test run, and it asserts that the saved model's thresholds match the frozen
values, failing loudly rather than silently re-tuning. The test split was read
once, after every modelling decision was fixed.