# Step notebooks

One notebook per pipeline step, in run order. Each one explains its step in numbered sections, and every code
cell is cut from the matching script in `src/`, so the notebook and the script cannot drift apart
(`python -m unittest "Final Project/tests/test_notebooks.py"` checks it).

| Step | Notebook | Script | Needs the raw files? | Writes |
|---|---|---|---|---|
| 1 | `Step1_Temporal_Analysis.ipynb` | `step1_preprocess_temporal.py`, `step1b_retention_cutoffs.py` | yes (957 MB user file) | `temporal_type_by_daybin.csv`, `temporal_summary.txt`, `retention_daily.csv`, `retention_cutoffs.txt` |
| 2 | `Step2_Build_Dataset.ipynb` | `step2_build_dataset.py` | yes (user file and 4.8 GB product file) | `candidates`, `history`, `sessions`, `meta_subset` (`.parquet`), `dataset_manifest.json` |
| 3 | `Step3_Feature_Engineering Final.ipynb` | `step3_features.py` | no, Step 2 tables | `features.parquet`, `feature_summary.json` |
| 4 | `Step4_Baselines_LambdaMART.ipynb` | `step4_baselines_lgbm.py` | no, Steps 2 and 3 tables | `lgb_model.txt`, `test_pred_lgb.parquet`, `results_baselines_lgb.json` |
| 5 | `Step5_Neural_Rankers_DHAR.ipynb` | `step5_neural.py` | no, Steps 2 and 3 tables (GPU recommended) | checkpoints, `test_pred_neural.parquet`, `results_neural.json` |
| 6 | `Step6_Diagnostics.ipynb` | `step6_diagnostics.py` | no, Steps 2 to 5 outputs | `segment_metrics.csv`, `paired_bootstrap.json`, `ablation.json`, ... |
| 7 | `Step7_ColdStart_Diversity.ipynb` | `step7_coldstart_diversity.py` | no, Steps 2 to 4 outputs | `results_coldstart_diversity.json`, `diversity_tradeoff.csv`, ... |

## How each notebook is organised

* **Part A** rebuilds the step (the same code as the script, split into explained cells).
* **Part B** reads the measured outputs of the corrected run from `artifacts/corrected` and re-checks them:
  the split fingerprint (Step 2), every baseline and neural metric recomputed from the saved predictions
  (Steps 4 and 5) and the paired bootstraps (Step 6). It needs no raw data.
* Part A only runs if its inputs exist and it would not overwrite `artifacts/corrected` (the `RUN_STEP`
  switch in section 1.1). Otherwise it is skipped and Part B still runs, so **Run All never fails**.

## Running

* **Colab.** Open a notebook, `BRANCH` in the first cell is already `main`, and set `MOUNT_DRIVE = True` if the raw files are on
  Google Drive (default folder `MyDrive/JDsearch/`). Outputs then go to Drive, so the next step finds them.
  Run Steps 1 to 7 in order to regenerate everything and keep the outputs in the notebooks.
* **Locally.** Set `JD_USER_TXT`, `JD_PRODUCT_TXT` and a new `JD_RUN_DIR` (so the committed evidence in
  `artifacts/corrected` is not overwritten), then open the notebook from `Final Project/notebooks/`.

## What the saved outputs in this copy are

All 7 notebooks, **Part A and Part B**, were executed locally on a MacBook Pro (Apple M4, MPS for
Step 5) on 7 November 2026 from commit `4a6e32d`. Random, Popularity, Content and LambdaMART reproduce
the `artifacts/corrected` NDCG@10 values exactly (train, validation and test); SVD matches to within
2e-5 NDCG@10 (nondeterministic randomized SVD), rounding to the same reported figure. Neural test
NDCG@10 differs by up to 0.0034 (Apple MPS is not bitwise deterministic). The report still uses the
figures in `artifacts/corrected`.

## Notes from building these notebooks

* **Split fingerprint was platform dependent.** The `split_sha256` in `dataset_manifest.json` was computed on
  Windows, where pandas writes CRLF line endings, so recomputing it on Linux, macOS or Colab gave a different
  hash for the identical split. `step2_build_dataset.py` now fixes the line ending (CRLF), so the stored hash
  stays valid everywhere; `Step2_Build_Dataset.ipynb` (B.2) recomputes it and shows `True`.
* **Step 1** is the teammate's notebook with four edits: the 7-30d click share now reads 0.0028% (it said
  0.000028%, a fraction printed as a percentage), a Colab-aware setup cell, unused variables removed, and no
  long dashes. Outputs of unchanged code cells are untouched; the setup cell's old output held a local path
  and was cleared.
* To regenerate a notebook after editing a script, edit the script first and rebuild the notebook from it; the
  tests fail if they disagree.
