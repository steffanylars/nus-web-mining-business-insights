# Step 7 - Cold-start routing and category diversification

Branch: `steff-coldstart-diversity` (on top of `ethan`). Date: 1 November 2026 (Singapore).

## Why this step exists

Two review comments on the draft:

1. *Cold-start users / users with short history are scored by the same model as warm
   users - they should be scored with current-query matching, metadata and training-set
   popularity instead.*
2. *No category diversification is applied after relevance scoring.*

Both are now implemented in `src/step7_coldstart_diversity.py`, registered as stage `7`
in `run_pipeline.py`, with regression tests in `tests/test_pipeline.py`, and as the executed
notebook `notebooks/Step7_ColdStart_Diversity.ipynb` (the same code, cell by cell, with explanations and outputs preserved). Both were then
**measured, and both selection rules (validation only) chose "do not apply"**. This file
records what was built, what the data says, and how it should be reported.

## What was built

| Component | Implementation | Selection (validation data only) |
|---|---|---|
| History-free fallback ranker | LambdaMART (same `LGB_PARAMS` as stage 4) on 10 features that do not use the user's history: `query_len, slate_size, has_meta, ov_name, ov_cat, ov_brand, jacc_name, pop_wid, pop_cate3, pop_brand` | - |
| Cold-start router | Sessions with `n_hist <= T` get the fallback score; others keep the full model's score (whole slate switches, so no cross-model calibration is needed) | `T` ∈ {never, 0, 1, 2, 3, 5, 10, 20}; max validation NDCG@10 |
| Category diversification | Greedy MMR over the top-10: `(1-λ)·rel − λ·(#already-selected items in the same cate_id_3)`; positions beyond 10 keep relevance order; λ = 0 reproduces the relevance ranking exactly | λ ∈ {0, .05, .1, .2, .3, .5}; largest λ whose validation NDCG@10 stays within 0.005 of λ = 0 |
| Diversity metric | Mean number of distinct level-3 categories in the top-10 (and the share of distinct categories) | reported, not optimised |

Outputs: `results_coldstart_diversity.json`, `coldstart_segments.csv`,
`coldstart_threshold_selection.csv`, `diversity_tradeoff.csv`,
`figures/fig_diversity_tradeoff.png`, `model/test_pred_step7.parquet`,
`model/lgb_fallback_history_free.txt` under the run directory.

## What the data says

**Provenance of these numbers.** Run on the *draft* pipeline's artifacts (Steff's
`artifacts/model/{features,sessions,meta_subset}.parquet` from the 26 Oct run). Same 20,000
users, same seed-42 70/15/15 split as the `ethan` branch (session ids and split membership
verified identical), but the draft feature values, not Ethan's corrected ones. The full model
was refit in step 7 because no stage-4 booster was present in that run directory (test NDCG@10
0.3159, consistent with the draft's 0.3125). Results are saved under
`artifacts/step7_draft_artifacts/`. **Rerun on the corrected artifacts** with
`python "Final Project/src/run_pipeline.py" --stages 7` once stages 2–4 of `ethan` have been run
locally; the conclusions below are expected to hold but must be re-measured.

### 1. Cold start: the full model already handles short histories better than a history-free scorer

NDCG@10 on test, by history length of the user (positive queries only):

| history_group   |   fallback_history_free |   full |   routed (T=1) |   positive_queries |
|:----------------|------------------------:|-------:|---------:|-------------------:|
| limited(1-2)    |                  0.2886 | 0.3090 |   0.2847 |                 63 |
| very_short(3-5) |                  0.3035 | 0.3493 |   0.3493 |                101 |
| short(6-20)     |                  0.3040 | 0.3541 |   0.3541 |                393 |
| medium(21-100)  |                  0.2672 | 0.3016 |   0.3016 |               1115 |
| long(>100)      |                  0.2658 | 0.3144 |   0.3144 |               1293 |
| ALL             |                  0.2731 | 0.3159 |   0.3154 |               2965 |

- The history-free ranker is **worse in every segment, including the shortest-history users**
  (0.289 vs 0.309 on 63 queries). The full LambdaMART already receives `h_len`, `h_n_*`,
  `h_min_elapsed` and the affinity features, which are simply zero/absent for short histories;
  it degrades gracefully instead of needing a separate model.
- Validation picked `T = 1` by +0.0003 NDCG@10 over "never route" - within noise - and on test
  routing those 35 sessions **hurt** the limited group (0.309 → 0.285). Paired bootstrap of
  routed − full on all test queries: −0.0005, 95% CI [−0.0016, +0.0004].
- There are **no zero-history users in JDSearch** (minimum `n_hist` = 1 in the sample); only 459
  of 20,000 users have ≤ 2 events. A true cold-start regime does not exist in this dataset, so
  the recommended fix answers a question the data does not pose.
- **Where cold start really lives is the item side**: in Ethan's corrected run, positives never
  seen in any training history reach the top-10 only 19% of the time versus 81% for seen items
  (`artifacts/corrected/positive_subgroup_full_slate.csv`, median full-slate rank 42 vs 4). That
  is the lever, and it is addressed by metadata features, not by user routing.

**Reporting recommendation.** Keep the full model for all users. Report this as a diagnostic
(Aspect 4): "a history-free fallback for short-history users was built and evaluated; it is
dominated by the full model in every segment because JDSearch has no zero-history users and the
feature ranker already conditions on history length."

### 2. Diversification: a pure relevance cost on this task

|   λ |   val NDCG@10 |   test NDCG@10 |   test MRR@200 |   test P@1 |   distinct cate_3 in top-10 (test) |
|------:|-------------:|--------------:|--------------:|----------:|---------------------------------:|
|  0    |       0.3117 |        0.3154 |        0.3271 |    0.2044 |                           2.19 |
|  0.05 |       0.3027 |        0.3064 |        0.3241 |    0.2044 |                           3.46 |
|  0.1  |       0.2926 |        0.2983 |        0.3212 |    0.2044 |                           4.14 |
|  0.2  |       0.2766 |        0.2835 |        0.3138 |    0.2044 |                           4.83 |
|  0.3  |       0.2652 |        0.2727 |        0.3073 |    0.2044 |                           5.14 |
|  0.5  |       0.2441 |        0.2526 |        0.2909 |    0.2044 |                           5.46 |

- Even the mildest setting (λ = 0.05) costs −0.009 NDCG@10 to gain +1.3 categories in the top-10;
  the selection rule therefore chose **λ = 0**. P@1 is untouched by construction (the top item is
  never displaced).
- This is expected for *search* re-ranking: the slate answers one query, and the relevant
  items are concentrated in one or two level-3 categories (2.2 distinct categories in the
  undiversified top-10 already reflects the query, not a model failure). Diversification is a
  product decision (exploration, exposure fairness for cold items), not a relevance improvement,
  and should be presented with its measured price, not as a free fix.

**Reporting recommendation.** Include the trade-off curve (`fig_diversity_tradeoff.png`) in the
business-insight section: "JD could buy ≈ +1.3 extra categories per top-10 for ≈ 3% relative
NDCG@10, which is a merchandising choice rather than a ranking improvement."

## Relation to the other AI-suggested items

| Suggestion | Status |
|---|---|
| Content projection for cold items | Already present: `ov_*`, `jacc_name`, `pop_cate3`, `pop_brand` and the category/brand embeddings in `step5_neural.py` score cold items from metadata |
| Shuffle slates against position bias | Already handled: scores are per candidate and order-invariant; ties use deterministic ID keys (`eval_utils.tie_keys`), so slate order cannot leak |
| `is_queryless_heavy` feature | `h_queryless` already exists as a continuous feature; a binary cut adds nothing a tree ranker cannot learn |
| Subword embeddings for item names | Not applicable: JDSearch names are anonymised term IDs, so there are no characters/subwords to share between items |
| Hard negative sampling | Not applicable: training is listwise softmax over each slate, which already weights the highest-scoring negatives most |
| Dual-horizon attention | Implemented in the draft; the matched comparison on `ethan` shows no gain over the same-feature MLP (see `ETHAN_CHANGES.md`) |

## Checks

- `tests/test_pipeline.py` adds `test_coldstart_routing_switches_whole_sessions` (routing is
  per session and `T = -1` is the identity) and `test_mmr_identity_at_zero_and_category_coverage_monotone`
  (λ = 0 reproduces the relevance order, the top item is never displaced, positions past the
  top-k keep relevance order, and distinct categories do not decrease). All 11 tests pass.
- Thresholds and λ are selected on validation only; test is touched once per configuration for
  reporting.
