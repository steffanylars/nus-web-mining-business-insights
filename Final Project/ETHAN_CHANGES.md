# Ethan's pipeline corrections

Branch: ethan. Date: 29 October 2026 (Singapore).
Original draft reviewed at commit 3d6b62b90fbdee102331f2aaf099be0ec52cb5e8.

## What I changed for the team

I corrected the pipeline issues found during the audit, reran the same 20,000-user
sample, and added matched architecture comparisons. The original sample and
train/validation/test membership are preserved. Main has not been edited.

The strongest change to our report's interpretation is that **the corrected
experiments do not show an advantage for dual attention beyond the same-feature MLP**.
We should present the temporal analysis and feature engineering as supported
contributions and discuss the architecture result honestly.

## Changes and reasons

| Area | Original behavior | Change | Why it matters |
|---|---|---|---|
| Tokenization | Split literal ^X or underscore | Split actual U+0018 consistently | Query/name/brand/category overlap now measures shared terms |
| Metadata | Omitted category name | Read complete raw metadata directly, including category names | Category overlap is no longer forced to zero; private chunk files are unnecessary |
| History | Keep only latest 60 events before all features/towers | Keep every preceding event for aggregates; sample each neural horizon independently | Old orders cannot be crowded out by recent clicks |
| Horizon definitions | Narrative and code differed; 90-day cart flag unused | Short = clicks <=7d or carts <=90d; long = all preceding orders | One explicit definition shared by features and neural packing |
| Neural history length | Latest 20 after shared cap | Up to 20 known-category positions spaced over each full horizon | Bounded inputs retain the temporal span; sampling coverage is recorded |
| CF baseline | Test users had untrained user-ID embeddings | GMF/MLP adaptation with a pooled preceding-item history encoder | Held-out users can be represented without using their current outcome labels |
| SVD | Fit shared factors on all split histories | Fit item factors on training histories; fold held-out histories into them | Clear training/inference boundary |
| Tie-breaking | Random keys depended on row traversal | Seeded keys from query/candidate identity | Candidate shuffling does not change a tied ranking |
| Model selection | LightGBM validation policy differed from final scorer | Custom validation NDCG uses the same positive-query policy | Checkpoint selection and final metrics agree |
| Ablation | Only a tree feature ablation; different full-model configuration | Add MLP, pooled, short-only, order-only and dual variants; match tree ablation configuration | We can test whether the sequence towers add value |
| Item diagnostics | Evaluate warm/cold products after shortening slates | Rank full slates first, then inspect positive subgroup rank/recall | Subgroups keep their actual competitors and disclose denominators |
| Reproducibility | Personal Mac paths, unstable brand hashes, state-only checkpoint | Central config, raw metadata reader, deterministic hash, saved preprocessing and scorer | Another teammate can reproduce the run and reload a model |
| Reporting | Hand-entered, inconsistent scores and strong causal claims | Generate corrected Markdown report/results from measured JSON | Tables stay consistent and conclusions match evidence |

The new CF baseline is a history-conditioned adaptation, not a faithful reproduction
of the original NCF paper. The original user-ID experiment should not be used to
claim that collaborative filtering inherently fails on JDSearch.

## What the rerun shows

See [corrected results](artifacts/corrected/RESULTS.md) for the complete table.

| Model or experiment | Original draft NDCG@10 | Corrected run NDCG@10 |
|---|---:|---:|
| Popularity | 0.2698 | 0.3045 |
| Content overlap | 0.0867 | 0.0951 |
| SVD | 0.2735 | 0.2759 |
| LambdaMART | 0.3125 | 0.3630 |
| DHAR | 0.3362 | 0.3713 |
| User-ID NCF / history-conditioned CF adaptation | 0.0858 | 0.2694 |
| Same-feature MLP without towers | Not tested | 0.3725 |
| Pooled-history control | Not tested | 0.3738 |

These are before/after pipeline comparisons, not controlled estimates of one
individual fix. Complete histories and several protocol changes affect the scores.

The temporal/action feature bundle adds **0.0243 NDCG@10** to LambdaMART under the
matched configuration (0.3387 without, 0.3630 with). That supports the bundle's value.

DHAR minus the MLP is **-0.00116**, with paired query bootstrap 95% interval
**[-0.00336, +0.00109]**. We cannot claim the attention towers improve on these
features in this run. The pooled model's test score is also higher than DHAR's;
we must not select the final model solely by comparing test results.
Short-only has the highest validation NDCG among the five tower comparisons in
this run; choosing model families and thresholds still belongs on validation data.

## Data and validation

- Exact original user sample and split preserved: 14,000 train, 3,000 validation, 3,000 test.
- Complete sampled histories now contain **3,091,153 events**, versus 907,554 after the old cap.
- Every sampled order is retained in the full history before per-horizon sequence selection.
- Full raw temporal analysis rerun: 173,831 users and 26,667,260 events.
- Nine regression checks cover delimiter parsing, elapsed-time alignment, old-order retention,
  explicit horizons, order-independent ties, positive-query validation, training-only SVD,
  held-out neural preprocessing, empty attention and full-slate subgroup ranks.
- Saved prediction/metric and standalone checkpoint-reload verification passed and is recorded in
  artifacts/corrected/verified_run.json (686 reloaded candidate scores, maximum error 1.2e-7).
- All models use the same full test candidate population: 266,640 rows and 2,965 positive queries.
  The other 35 test queries have no positive outcome and are excluded from ranking averages.

This is one training seed on the existing test partition, which had already been
inspected in the draft. Bootstrap intervals reflect query resampling for fixed models,
not variability across training seeds or an untouched final holdout.

## Files to open

1. This file for the plain-language change log.
2. [REPORT.md](artifacts/corrected/REPORT.md) for the corrected technical report draft.
3. [RESULTS.md](artifacts/corrected/RESULTS.md) for generated measured tables.
4. [README.md](README.md) for reproduction and inference commands.
5. [The original audit](../audit/PIPELINE_REVIEW.md) for the original issues and evidence.
6. The changed src/ files and tests/ for implementation details.

Corrected outputs live under artifacts/corrected. Original report/, slides/ and
root artifacts/ files are retained as historical draft snapshots. They do not
describe the corrected code. The new report is Markdown for review; this branch
does not silently replace the team's existing DOCX/PDF or presentation.

Large raw files, regenerable intermediate tables, SVD state, neural checkpoints
and the large preprocessing item map remain local and are git-ignored.
Prediction evidence, measured summaries and lightweight configuration are committed.

## What still needs a team decision

- Confirm that supervised reranking on user-split current-query labels matches the course requirements.
- Decide whether the architecture complexity is justified and which model to present as primary,
  using validation evidence.
- Consider a fresh final holdout or repeated seed/split evaluation before stronger generalization claims.
- Check the official deliverable guideline and add the team's real contribution and AI-use statements.
- Format the agreed report and slides for submission. The archived original claims should not be reused unchecked.

Suggested message to the team:

“I pushed my pipeline corrections to ethan. The change log is Final Project/ETHAN_CHANGES.md.
The temporal pattern reproduces, but I fixed token parsing, restored complete histories,
repaired the held-out-user CF setup and added matched ablations. The corrected scores
improve, while dual attention does not beat the same-feature MLP in this run. I kept
the original documents for comparison and put corrected evidence/report text in
artifacts/corrected. Please review the changes before we decide the final report story.”
