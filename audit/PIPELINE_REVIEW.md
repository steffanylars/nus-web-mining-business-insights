# JDSearch final pipeline review

Historical audit of the original draft. The implemented corrections and rerun are documented in [ETHAN_CHANGES.md](../Final%20Project/ETHAN_CHANGES.md).

Reviewed 29 October 2026 at repository commit `3d6b62b90fbdee102331f2aaf099be0ec52cb5e8`.
Local branch: `codex/final-pipeline-audit`. No production code, reports, remote branches or GitHub discussions were changed.

The draft has a useful temporal finding and its saved neural scores reproduce. It is not ready to be treated as a validated final pipeline: content features are incorrectly parsed, history truncation undermines the long-term representation, and the NCF experiment cannot support its stated conclusion. Resolve these before rewriting the report around the results.

## What was checked

- Read every modeling stage, evaluation utilities, results tables, report generator and extracted text of the committed final DOCX. This is a technical/content review, not a rendered layout review.
- Independently scanned all 173,831 raw behavior records and reconstructed all 26,667,260 history-event ages. The temporal composition counts match the committed CSV exactly.
- Reconstructed the seeded 20,000-user sample: 1,781,978 candidates and 907,554 retained history events, matching the draft.
- Compared the tokenizer against raw query/name data for 200 sampled query records, scanning the raw metadata catalogue for their candidates. This is a diagnostic sample, not a full-data content experiment.
- Recomputed scores from both committed prediction files: 266,640 test candidate rows, identical keys/labels, no duplicate keys, all 3,000 test IDs represented. There are 2,965 queries with a positive label.
- Did not retrain models. Intermediate features, metadata chunks and neural preprocessing state are not committed. The deliverable guideline itself was not available in this repository; this review does not certify rubric compliance.

Evidence: `verification.json`. Reproduction code: `verify_pipeline.py` (requires numpy, pandas and pyarrow, plus the local raw files identified in the script).

## 1. Fix query and metadata tokenization before interpreting content results

**High priority.** `Final Project/src/step3_features.py:22–25` splits literal `^X` or `_`. JDSearch uses the actual control character `\x18` within term sequences. `_` separates history events and candidate lists; it is not the term delimiter. A raw query `12\x1832\x1856` currently becomes one token rather than three.

Independent diagnostic, using the first 200 records by source index within the same seeded sample:

| Measure | Current tokenizer | Documented delimiter |
|---|---:|---:|
| Candidate pairs with metadata | 18,140 | 18,140 |
| Pairs with nonzero query/name overlap | 0 | 16,994 |
| Mean overlapping unique terms per pair | 0.000 | 1.962 |
| Mean unique query terms, weighted by matched candidate pairs | 1.000 | 2.707 |

The zero-gain `query_len`, `jacc_name` and `q_hist_overlap` features are consistent with this defect. Anonymization preserves equality of shared term IDs; it does not make token-overlap relevance intrinsically ineffective.

**Action:** tokenize query, historical query, product name, brand name and category names consistently using `\x18`. Add a small delimiter/alignment fixture. Rebuild features and rerun content, LambdaMART and DHAR. Withdraw the claim that the 0.087 content score establishes a weak content signal until then. This defect affects all feature-consuming models, not only the content baseline.

**Related missing field:** step 2 selects category IDs but omits `cate_name_3` from the metadata projection. Step 3 therefore always creates an empty category-name token map; `ov_cat` is always zero. Include the category name explicitly if keeping that feature.

## 2. Construct horizon representations before imposing a shared history cap

**High priority.** `step2_build_dataset.py:36,75–78` keeps the 60 most recent events across all action types. Step 3's order profile and step 5's long sequence are built only after that deletion. Step 5 then limits each horizon to 20 categories. The long-term tower does not receive the full order history.

Reconstruction of the exact 20,000-user sample:

| Quantity | Before cap | After cap |
|---|---:|---:|
| Order events | 785,868 | 128,841 |
| Order events older than 90 days | 550,681 | 60,274 |
| Users with at least one order | 18,310 | 15,391 |

The cap removes **83.6% of orders** and **89.1% of orders older than 90 days**; 2,919 users lose every order. This discards precisely the information motivating the proposed architecture.

**Action:** reconstruct ages on complete histories; calculate full-history aggregates; select recent and order sequences separately, each with an explicit bounded sampling rule. Record retained event coverage per horizon. If limiting long histories, validate that choice rather than calling it complete long-term history.

Also reconcile the definitions: both short profiles and neural short sequences currently use **all events within seven days**, including orders/follows, rather than only recent clicks/carts. `recent90` is computed but never used. State whether 7–90-day carts form a separate channel or are intentionally excluded from the short sequence. `h_min_elapsed` is the age of the user's most recent retained event, not candidate-specific last interaction age; fix its report description or implement the candidate-specific feature.

## 3. Repair the NCF baseline and align the SVD protocol

**High priority.** `step5_neural.py:178–200` learns session-ID and item-ID embeddings from current candidate labels for training sessions only. Since users/sessions are disjoint across splits, every test user embedding receives no training updates. The model never uses the test user's available historical interactions to infer preferences. A near-random score is therefore expected from this particular setup; it does not demonstrate that collaborative filtering intrinsically fails on JDSearch, nor that unseen queries are the cause. NCF does not consume queries at all.

**Action:** use a history-conditioned representation that can score held-out users, or define a legitimate adaptation protocol using their preceding histories without accessing current candidate outcomes. At minimum, present the existing run as a deliberately unsupported user-ID baseline, not evidence against the whole CF family. Include training/validation performance and item-coverage breakdowns if retaining it.

`step4_baselines_lgbm.py:49–58` fits SVD on histories from **all** splits, whereas popularity is fit from training histories. Using a test user's preceding history at inference is allowed; jointly fitting the shared factorization on every test history is a transductive protocol. It is not current-label leakage, but it differs from training-only representation fitting.

**Action:** fit shared item factors from training histories and fold held-out users into that fixed space, or explicitly label and justify the transductive variant. Do not describe SVD as NMF: this code uses TruncatedSVD, not nonnegative matrix factorization.

## 4. Keep the temporal finding, qualify the retention explanation

**Strong finding, overstated explanation.** The suffix-sum indexing is correct. For the documented `[0,12,4,5]` gaps, event ages are `[21,9,5]` seconds before the query. All raw temporal counts reproduce.

The history composition changes sharply around seven and 90 days. But the data contain 107 clicks aged 7–30 days, 1,203 carts aged 90–180 days and one cart beyond 365 days. No order falls in the >=365-day bin. Daily bins cannot establish an exact second-level cutoff, and these distributions alone do not establish JD's actual storage policy.

**Suggested wording:** “The sharp, action-specific drops are consistent with differential logging or retention windows: roughly one week for clicks and three months for carts, while orders extend much further back. We therefore treat missing older clicks as unobserved, rather than evidence of no interest.”

Avoid “exact cutoff,” “JD keeps,” “clicks vanish,” and “orders >=1 year” unless independently documented. Distinguish a plausible logging mechanism from a verified platform policy.

Historical action/age confounding does **not** invalidate ordinal modeling of the separate current candidate outcome. Current-label gains `[0,1,3,7]` are an explicit utility choice; discuss their rationale separately. The AI-log rejection of ordinal modeling on the basis of historical retention conflates inputs with targets.

## 5. Saved ranking metrics reproduce, but architecture attribution is untested

| Saved prediction | Recomputed NDCG@10 | MRR@200 | P@1 |
|---|---:|---:|---:|
| LambdaMART | 0.312468 | 0.324612 | 0.204047 |
| DHAR | 0.336156 | 0.351088 | 0.229342 |
| NCF | 0.085755 | 0.104314 | 0.032715 |

These match the JSON results. Standard three-decimal rounding of LambdaMART is **0.312**, whereas the prose repeatedly says 0.313. README also has an older DHAR headline (0.335/0.350/0.225); regenerate all tables and prose from one results source.

A paired query bootstrap of the saved predictions gives DHAR minus LambdaMART **+0.02369 NDCG@10**, with a 95% interval **[+0.01690, +0.03010]** (2,000 resamples, seed 42). This is evidence for a difference between these particular saved models on this sample. It does not validate broken features, establish stability across training seeds, or identify why the models differ.

The current ablation removes a bundle of temporal/action features from LambdaMART, not either DHAR tower. It supports a bundle-level result (0.29541 versus 0.28225, delta 0.01315). It does **not** establish that dual attention explains DHAR's improvement. The ablation's full model also uses a different training configuration from headline LambdaMART (400 vs 600 maximum rounds; no row/feature subsampling).

**Action:** after correctness fixes, compare DHAR with the same numeric-feature MLP without histories, a pooled-history encoder, short-only, order-only and both horizons. Match data, sampling and training budget and select using validation data. Report uncertainty across seeds or query bootstrap as appropriate. Call the existing LambdaMART experiment “temporal/action feature bundle ablation.”

## 6. Interpret diagnostic populations correctly

`step6_diagnostics.py:71–76` removes cold candidates before evaluating warm candidates, and vice versa. It then drops any query with no positive left in that subset. The 0.586 versus 0.180 comparison therefore uses different candidate pools, ideal rankings and often different queries. It is a valid within-subset descriptive metric if labeled precisely, but **not** a full-slate cold-item performance comparison or an upper performance ceiling.

**Action:** retain the original ranked slate for cold-positive recall/rank diagnostics; state the relevance population and denominator. Separately report candidate training-history coverage and current training-label exposure. `pop_wid == 0` means absent from the **capped training histories**, not necessarily unseen in supervised training candidate rows or their positive labels.

The 80.54% number is the share of distinct candidate products absent from all supplied histories; it is not a percentage of the complete catalogue or a direct training cold-start rate. MF's warm coverage uses a different universe again (all split histories). Give each number its actual denominator.

The purchase/click median-rank difference is measured for LambdaMART, not DHAR. LambdaMART explicitly optimizes larger purchase gains, and different query/product groups can differ. It cannot prove that old order history causes better purchase prediction or validate the two-horizon hypothesis. Likewise, a smaller train/test gap alone does not explain why a model generalizes better.

Users with 1–2 actions are a limited-history group, not observed zero-history users. All reported users have some history. A zero-history fallback requires a separate evaluation with history withheld.

## 7. Make reproduction and inference concrete

- All stages have absolute paths belonging to the original author's Mac. Metadata chunks are required but neither committed nor built by a supplied conversion stage. Fixing `N_USERS` alone will not make this pipeline run elsewhere.
- The neural device selects MPS or CPU, never CUDA. A Colab GPU run will use CPU as currently written.
- `hash(brand)` is process-randomized unless Python's hash seed is fixed before launch. NumPy and torch seeds do not stabilize it. Use a deterministic brand mapping/hash and save it.
- The checkpoint stores only `state_dict`. Reusable inference also needs category/brand mappings, feature order, normalization statistics, architecture settings and history-selection rules. The model class is local to `main()` and there is no independent inference entry point.
- Pin dependency versions and centralize paths/configuration. Add the raw-to-metadata-chunk step or read raw metadata directly. Preserve split membership and raw-row IDs.
- Evaluation random tie-breaking fixes the serious source-order advantage for a fixed input order. Its random keys are assigned by row traversal, so arbitrary row permutations still change the realization. For reproducible comparisons, assign deterministic random keys to `(seed, query ID, candidate ID)` or average several tie realizations; never use source position.
- The custom evaluator excludes all-negative queries. Align validation model selection with that policy (including LightGBM's treatment of zero-IDCG groups) and retain the excluded count. State that this project trains on a user split of current-query candidate outcomes, which differs from the original JDSearch history-training/last-query-test benchmark.

## Recommended sequence for the team

1. Agree the task: supervised reranking of provided slates with disjoint users, graded current outcomes, and macro query metrics. Keep all candidate outcomes isolated from features and preprocessing fitted on training data.
2. Correct delimiter/category metadata and build full-history aggregates plus independently sampled horizons. Freeze the split; run cheap feature sanity checks before training.
3. Rerun random, training-history popularity and corrected content; then LambdaMART. Repair or explicitly bound the CF baselines.
4. Train DHAR with matched MLP/history ablations. Use validation to choose settings; do not repeatedly tune against the current test set. Consider a fresh untouched final holdout if test results have already guided design choices.
5. Recompute full-slate subgroup diagnostics, gains, uncertainty and all tables; save preprocessing with models. Rewrite report claims to match the resulting evidence.

## Suggested points to raise with the team

“The temporal finding reproduces and is a good basis for the project. Before finalizing results, I found three implementation issues we should resolve: the raw term delimiter is not parsed, the 60-event cap drops most older orders before the long-term tower sees them, and NCF uses untrained held-out user embeddings. The saved DHAR metrics reproduce, but we need corrected features and tower ablations before attributing its gain to dual horizons. I have kept this as an audit on my own local branch.”

## Sources

- [Pinned project source](https://github.com/steffanylars/nus-web-mining-business-insights/tree/3d6b62b90fbdee102331f2aaf099be0ec52cb5e8/Final%20Project/src)
- [Official JDSearch field definitions and delimiters](https://github.com/rucliujn/JDsearch#dataset-files)
- Local raw data independently scanned; no external platform retention policy was verified.
