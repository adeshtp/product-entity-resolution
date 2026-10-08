# Interview notes

Questions worth being able to answer about this project, with the reasoning rather
than a restatement of the README. Numbers refer to the test split unless stated.

## Why is exact matching not enough?

Only 10 of the 1,097 gold pairs have identical names after normalization. The two
catalogs describe the same product differently: Abt writes "Sony Turntable -
PSLX350H / Belt Drive System", Buy writes "Sony PSLX350H Turntable, Belt Drive".
On my validation split the exact-match baseline matched **0 of 161** products,
because that split happens to contain none of those 10 pairs. So the model number
survives but the surrounding words do not, which is why I use character n-grams
rather than word tokens: character fragments survive reordering and truncation in
a way whole words do not.

## Why block at all?

The full grid is 1,081 × 1,092 = 1.18 million pairs and only 1,097 are matches.
Scoring all of them is slow and pointless. Retrieval keeps 1.83% of the grid and
drops **no** gold match at k=20, so the reduction is free. The important part is
reporting it honestly in both directions: candidate Recall@20 = 1.0000 *and* zero
matches lost. If retrieval had lost matches, a good matcher would look worse for
no fault of its own, which is why I report the two stages separately.

## How did you avoid training on data you later evaluate on?

Everything fitted is fitted on train only: the retrieval TF-IDF, the two feature
TF-IDF objects, and the scaler, which sits inside the Pipeline so it only ever
sees training rows. Negative mining is the subtle one — candidates for validation
are drawn from the *whole* Buy catalog, because that is the real task, so if I had
mined negatives against all of Buy the model would have trained on records the
evaluation uses. Negatives therefore come from train A × train B only, and the
pipeline raises at runtime if any training pair points outside that pool.

The split is on connected components of the gold graph rather than rows, because
the mapping has 21 non-1-1 groups. A row split would put one side of a 2A+1B pair
in train and the other in test.

## What are hard negatives and why bother?

Random negatives teach nothing, because a random catalog pair shares no words.
The pairs that actually cause errors are same-brand, similar-wording,
similar-price products that are nonetheless different. I mine those from the
retrieved candidate list (14,354 of them) and add a smaller set of easy negatives
sampled from outside it, so the model also learns that "retrieved" does not mean
"probably a match".

## Why Logistic Regression?

Three reasons, all about being able to defend it. Interpretability first: I can
print the coefficients, and they are the single most useful artifact in the
project. `name_similarity` +3.19, `price_difference` −1.42,
`model_token_conflict` −0.74. That table predicted the dominant failure mode
before the error analysis confirmed it. Second, proportionality: 18,900 rows and
10 features is a small dense tabular problem. Third, calibration: the sigmoid
output is directly usable as a threshold, which is what a three-way decision
policy needs.

I report these as model scores, never as confidence, because I did not evaluate
calibration.

## Why does the system produce REVIEW?

Because the errors are not symmetric. Merging two different products corrupts a
catalog invisibly; failing to merge one leaves two correct pages. So I refuse to
decide when uncertain: only the top-scoring candidate per product can be merged,
and anything between the two thresholds goes to a human. On test, 65 of the 74
missed gold pairs went to REVIEW and only 9 were rejected, so the strict
threshold is mostly deferring rather than discarding. Both recall figures are
reported: 0.5488 automatic, 0.9451 with a human.

## How were the thresholds chosen, and could that have been gamed?

The rule was written down before I looked at validation: the lowest threshold at
which MATCH precision reaches 0.95. I made a real mistake here worth
admitting — my first implementation swept a fixed 0.05 grid capped at 0.95, the
same value as the target, so the search could never satisfy its own rule and it
reported the target as unattainable when it was reachable at 0.99. The fix was to
search every unique validation score, not to change the target.

The honest caveat is the more interesting lesson. Selecting the *lowest*
qualifying threshold means landing on a point with no margin: validation precision
was 0.9623, only 1.2 points of headroom. On test it did not hold, and precision is
0.8571 with a CI of [0.7857, 0.9214], entirely below target. A precision target
measured on 166 validation pairs is not a guarantee on unseen data. The
disciplined response was to report that rather than re-tune, because the test
split was quarantined for exactly that reason.

## What is a false merge, and how many did you make?

A MATCH on two records that are not the same product — the most damaging error
here, because in a real catalog it fuses one product page into another and stays
invisible until someone checks the photos. I report it separately from the generic
false-positive count because that count is diluted: 15 false merges against 3,101
true negatives gives precision 0.8571, a number shaped mostly by easy negatives.

15 on test, 4 on validation, and the same failure mode both times: same brand,
different model number. Several distinct Panasonic DECT phones collapsing onto one
model, or a 52-inch TV matched to a 42-inch one.

## Why is your recall so low?

0.5488 by construction. The precision-first threshold refuses to merge anything
uncertain, and a wrong merge costs more than a wrong split. Of the 74 missed gold
pairs, 65 were queued for review rather than dropped, so a human clearing the
queue would recover to 0.9451. I would rather quote 0.5488 honestly than 0.9451
and let the reader assume it was automatic.

## What is your biggest limitation?

**This benchmark cannot measure real NO MATCH performance.** Every one of the
1,081 products has a gold match, so there is no genuinely unmatched record to test
against. Any "NO MATCH precision" I reported would really be measuring the
retrieval negatives, which are easy and chosen by me. So I make no such claim. The
closest I have is a probe that removes a product's gold partner and checks whether
the system still invents a match: 30 of 164 still had a candidate above the
threshold. That is a diagnostic, and it was used to select nothing.

Second is the precision gap itself — 0.9623 on validation against 0.8571 on test.
Retrieval held up (Recall@20 was 1.0000 on both), so the drop is in the matcher,
and with 164 test products the intervals are wide.

## What would you fix first?

A feature that separates product *lines* within a brand. The coefficients say the
problem is that `model_token_conflict` is too weak to veto `name_similarity`, and
it fails structurally in two observed shapes: `Weber Spirit SP-320` and
`Weber Q 320` both reduce to the model token `320`, so they *overlap* and the
conflict flag never fires; and an LG microwave with no model number in its title
gives the flag nothing to conflict with. I tried the cheap version of this fix —
mining extra same-brand negatives — and rejected it, because it increased false
merges from 10 to 12, the exact failure it targeted. Its apparent F1 gain came
only from the threshold rule re-selecting a looser operating point, which is not
a modelling improvement.

After that: nested cross-validation for the threshold, so the operating point is
averaged across folds with an uncertainty estimate instead of chosen once on 166
pairs.

## Where would you start in an interview?

By being precise about which question each number answers. Candidate Recall@K,
pair precision, Hits@K, MATCH-only recall and MATCH+REVIEW coverage are five
different questions, and collapsing them is how entity-resolution results get
overstated. I caught myself doing it: my first product-level metric was ranked by
retrieval order, which made Hits@K bit-identical to candidate Recall@K and
measured nothing about the matcher. Reporting both rankings separately is the fix.