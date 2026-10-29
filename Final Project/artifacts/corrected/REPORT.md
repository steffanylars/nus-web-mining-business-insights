# Personalised product search on JDSearch

## Corrected experiment status

This is the corrected technical report draft for team review on the ethan branch.
It supersedes the quantitative claims in the original report, slides and root artifact snapshots.
It is generated from the corrected run's JSON files rather than manually entered scores.
The original documents remain available for comparison; this draft still needs the team's contribution
statement and a check against the actual project deliverable guideline.

## Business objective and prediction task

For a user's current search query, rank the supplied candidate products using query terms,
product metadata and that user's preceding interactions. The output is an ordering of the
logged slate, not catalogue-wide retrieval or an estimated conversion funnel.

Candidate labels are 0 for no recorded interaction, 1 for click, 2 for cart and 3 for purchase.
Ranking utility uses gains [0,1,3,7]; this is a modeling choice that emphasizes purchases,
not a measured revenue function. Historical action types are inputs and are distinct from
these current outcome targets. No current candidate labels enter features.

## Temporal analysis and response to interim feedback

Each row's history interval list contains one more entry than its history events.
For event i, elapsed time before the current query is the suffix sum of gaps after i.
For [0,12,4,5], the three event ages are [21,9,5] seconds. The complete raw-data scan covers
173,831 users and 26,667,260 events.

Clicks become extremely rare after roughly seven days; carts become extremely rare after
roughly 90 days, while orders extend much further back. Rare exceptions remain: 107 clicks
occur in the 7-to-30-day bin, 1,203 carts in the 90-to-180-day bin and one cart beyond 365 days.
There are no orders aged at least 365 days in the supplied data. This pattern is consistent
with action-specific logging or retention windows, but does not establish JD's actual storage policy.

The implication is that absent older clicks are unobserved, not evidence of no interest.
Historical action type and recency cannot be interpreted independently as preference intensity.
This does not prevent using ordinal gains for the separate current candidate outcomes.

## Dataset preparation and features

The seeded sample contains 20,000 users, 1,781,978
candidate pairs and 3,091,153 complete historical events.
The 70/15/15 split keeps all candidates for one user together and preserves the original
sample and split membership. Raw source-row IDs and a split hash are saved.
This is supervised reranking trained on some users' current-query labels, which differs
from the original JDSearch benchmark's history-training/last-query-test protocol.

Query, historical query, name, brand name and category name fields use the actual control
character delimiter, U+0018. Category-name metadata is explicitly retained. Candidate metadata
coverage is 99.55% by candidate pair.

All history events contribute to aggregate profiles and training-history popularity.
A short intent channel contains CLICK events no older than seven days and CART events no
older than 90 days. The order channel contains preceding ORD events of any observed age.
The old global 60-event cap is removed. Numeric features include query/content overlap,
training-history item/brand/category popularity, horizon affinity, recency decay, exact-item
re-engagement and user activity. The minimum event age describes the user's last event,
not the candidate's last interaction.

Neural sequences are sampled independently from each complete chronological horizon using
20 evenly spaced known-category positions, retaining endpoints when the horizon is longer.
Missing or training-unseen categories map to zero and are excluded from sequence sampling.
The order channel has 785,868 eligible raw events and
237,250 selected known-category events across all splits.
This is a bounded representation of the full order horizon, not attention over every event.

## Models and training protocol

Random and training-history popularity establish simple reference points. Content ranking
uses query overlap with name/category/brand terms and name-token Jaccard similarity.
SVD fits shared item factors on training histories and folds each held-out user's preceding
history into those fixed factors. It does not fit shared factors on test users.

LambdaMART uses graded candidate labels and the engineered features. A custom validation
metric matches final evaluation's positive-query policy and deterministic tie keys.
DHAR combines candidate category/brand embeddings, separate attention over short and order
category sequences, and numeric features in an MLP. Matched variants disable both towers,
pool the histories, or enable only one tower. The MLP control still contains the same numeric
history features, isolating the incremental contribution of the sequence towers.

The history-conditioned CF baseline adapts the GMF/MLP approach with a pooled preceding-item
history encoder. It is not a faithful reproduction of the original user-ID NCF model.
It can represent a held-out user without an untrained session-ID embedding.
All neural variants use the same fixed training sample of up to 64 candidates per query (all positives retained), gain-based listwise loss,
optimizer, maximum epoch budget and validation early stopping. Shared item/category vocabularies
and numeric normalization are fit from training data. Brand hashing is deterministic.

## Evaluation

Metrics are averaged over complete candidate slates of queries containing at least one positive.
The fixed test partition has 3,000 queries, of which 2,965 are evaluable; 35 all-negative queries
are excluded from ranking averages. NDCG@10 uses exponential gains; MRR@200 and P@1 use label > 0.
HR@10 records whether any positive appears in the first ten positions.

Candidate IDs determine seeded tie keys independently of input row order. Source list position
is not a feature. Validation chooses checkpoints; the existing test partition is used for
reporting only in this corrected run.

| Model | NDCG@10 | MRR@200 | P@1 | HR@10 |
|---|---|---|---|---|
| Random | 0.0829 | 0.1034 | 0.0314 | 0.2277 |
| Popularity | 0.3045 | 0.3113 | 0.1865 | 0.5771 |
| Content/Relevance-only | 0.0951 | 0.1088 | 0.0354 | 0.2442 |
| SVD(train-only fold-in) | 0.2759 | 0.2819 | 0.1592 | 0.5356 |
| LambdaMART(features) | 0.3630 | 0.3669 | 0.2297 | 0.6516 |
| MLP(no towers) | 0.3725 | 0.3840 | 0.2519 | 0.6580 |
| Pooled history | 0.3738 | 0.3844 | 0.2513 | 0.6631 |
| Short only | 0.3713 | 0.3826 | 0.2503 | 0.6583 |
| Order only | 0.3731 | 0.3825 | 0.2496 | 0.6621 |
| DHAR(ours) | 0.3713 | 0.3831 | 0.2499 | 0.6583 |
| History-conditioned CF | 0.2694 | 0.2711 | 0.1403 | 0.5474 |

These figures replace the draft's results. Differences from the old run combine several fixes
and protocol changes and cannot be attributed to a single code correction.

## Architecture and feature ablations

| Architecture | Validation NDCG@10 | Test NDCG@10 |
|---|---|---|
| MLP(no towers) | 0.3676 | 0.3725 |
| Pooled history | 0.3679 | 0.3738 |
| Short only | 0.3692 | 0.3713 |
| Order only | 0.3678 | 0.3731 |
| DHAR(ours) | 0.3689 | 0.3713 |

DHAR differs from the same-feature MLP by -0.0012 NDCG@10 (paired query bootstrap 95% interval [-0.0034, +0.0011]). This run does not establish an advantage for dual attention beyond the engineered features.

The temporal/action feature bundle changes LambdaMART NDCG@10 from
0.3387 without the bundle to 0.3630
with it (difference +0.0243). Both arms use the same training configuration;
the full arm is the headline model. This tests the bundle, not the neural architecture.

## Diagnostic interpretation

History-length segments retain complete candidate slates. Limited-history users are described
as users with few observed actions, not verified new customers. Natural zero-history
performance is not claimed.

Item diagnostics rank each entire slate first, then measure ranks and recall of positives
present or absent in complete training histories. A second breakdown checks whether an item
appeared in any training candidate row. These are different exposure definitions.
The diagnostic CSV records the number of relevant items and queries for each subgroup.
It does not compare artificially shortened warm and cold slates or claim a cold-item ceiling.

Positive ranks by outcome type are descriptive and model-specific. A lower purchase rank does
not prove that old order history caused the difference, particularly with purchase-weighted gains.
The file rank_by_type.csv identifies both model and label.

## Limitations and team decisions

This run uses one sample and training seed. Query bootstrap uncertainty is conditional on the
saved models and does not estimate variation across training seeds. The original test set had
already been inspected during the draft and audit; it is not a new untouched final holdout.
A final independent holdout or repeated split evaluation would strengthen generalization claims.

Anonymized term equality supports overlap features but not semantic interpretation. Logged
noninteraction is not proof that an item was examined and rejected. Offline graded ranking
does not establish online conversion, revenue improvement or causal effects. Sequence lengths,
time windows, utility gains and sample scale remain modeling choices requiring sensitivity
analysis for stronger claims.

The team should confirm the final objective and approved evaluation protocol with the course
guideline, discuss whether the dual-tower complexity is justified by the measured ablations,
and supply its actual contribution and AI-use statements. No contribution assignments or
invented interaction histories are added here.

## Reproduction and saved inference

Use the documented environment variables and src/run_pipeline.py. Raw metadata is read directly;
prebuilt private chunks are unnecessary. New outputs live under artifacts/corrected.
The run preserves split/source identities, parameters, preprocessing statistics, vocabularies,
deterministic hashes, horizon coverage, model checkpoints and prediction files. Large data and
checkpoints are regenerated locally and excluded from Git. The saved-checkpoint inference entry
point scores prepared candidate features plus preceding histories without using current labels.

## References

- Liu et al. (2023), JDsearch, SIGIR. https://doi.org/10.1145/3539618.3591900
- Official dataset format and anonymized term delimiter. https://github.com/rucliujn/JDsearch
- He et al. (2017), Neural Collaborative Filtering, WWW. The history-conditioned baseline is an adaptation.
- Burges (2010), From RankNet to LambdaRank to LambdaMART.
- Ke et al. (2017), LightGBM, NeurIPS.
