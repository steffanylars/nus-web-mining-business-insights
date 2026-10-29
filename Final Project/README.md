# JDSearch corrected final-project pipeline

**Start with [ETHAN_CHANGES.md](ETHAN_CHANGES.md)** for what changed and why.
This branch is Ethan's proposed correction to the team draft. The original report/PDF,
slides, model files and tables outside artifacts/corrected are historical snapshots
and should not be submitted as results of the corrected code.

Corrected results and the reviewable report narrative are generated in
[artifacts/corrected/RESULTS.md](artifacts/corrected/RESULTS.md) and
[artifacts/corrected/REPORT.md](artifacts/corrected/REPORT.md).
The report still needs the team's contribution statement and rubric check.

## Task and protocol

Supervised reranking of the provided candidate slate for held-out users. Inputs are
current query terms, product metadata and the user's preceding history. Targets are
current labels 0/1/2/3, with ranking utility gains [0,1,3,7]. This differs from the
original JDSearch history-training/last-query-test benchmark.

The default run samples 20,000 users with seed 42 and preserves the draft's 70/15/15
user split. All preceding history events are retained for features. Neural short
sequences use clicks <=7 days or carts <=90 days; order sequences use all preceding
orders. Each horizon is independently sampled to 20 known-category positions across
its complete chronological span. Full candidate slates are evaluated, with
order-invariant deterministic tie keys; all-negative queries are excluded from macro
ranking averages and counted separately.

## Reproduce

Validated on Python 3.12.14, Windows CPU. CUDA, Apple MPS and CPU are selected in that
order when available. Install dependencies in your own environment:

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install -r "Final Project/requirements.txt"

Download the two raw JDSearch files using the [official documentation](https://github.com/rucliujn/JDsearch).
There is no dependency on the original author's private metadata chunks.

PowerShell, from repository root:

    $env:JD_USER_TXT = "D:\JDsearch\user_behavior_data.txt"
    $env:JD_PRODUCT_TXT = "D:\JDsearch\product_meta_data.txt"
    python "Final Project/src/run_pipeline.py" --users 20000

POSIX shells:

    export JD_USER_TXT=/data/JDsearch/user_behavior_data.txt
    export JD_PRODUCT_TXT=/data/JDsearch/product_meta_data.txt
    python "Final Project/src/run_pipeline.py" --users 20000

Or place the files in Final Project/data/. JD_RUN_DIR optionally selects a different
output directory; the default is Final Project/artifacts/corrected.
JD_THREADS, JD_SEED, JD_HISTORY_LENGTH, JD_EPOCHS and JD_BATCH_SIZE are configurable.
Change the run directory for a new experiment so outputs cannot be mixed.

Stages: 1 temporal bins; 1b daily age counts; 2 dataset and fixed split; 3 features;
4 baselines and LambdaMART; 5 six neural variants; 6 diagnostics and feature ablation;
report measured Markdown narrative. Select a subset with --stages, for example:

    python "Final Project/src/run_pipeline.py" --stages 4 5 6 report

Do not run later stages until their inputs are complete. Stage 5 writes incremental
results during training; rerun stage 6/report after all requested variants finish.
It evaluates one seed by default. Additional seeds are experiments, not a reason
to select the best test score. Checkpoints are selected by validation NDCG only.

## Checks

    python -m unittest discover -s "Final Project/tests" -v
    python audit/check_corrected_run.py

The second command requires the completed 20k corrected run and local checkpoints.
It verifies fixed split membership, full-history retention, all saved neural metrics,
candidate/label integrity and independent saved-model inference.
The original audit is in audit/PIPELINE_REVIEW.md. Its verifier reads the pinned
original source from Git so it remains valid after these changes.

## Saved inference

Preprocessing and checkpoints are generated together in artifacts/corrected/model.
The large metadata/history/features, item maps, SVD state and neural checkpoints
remain local and are git-ignored. Small measured tables, prediction files and run
verification are committed so the team can inspect the evidence.

predict_neural.py takes **prepared numeric candidate features** (including
training-only popularity priors), preceding histories and product metadata:

    python "Final Project/src/predict_neural.py" --checkpoint "Final Project/artifacts/corrected/model/dual_seed42.pt" --preprocessing "Final Project/artifacts/corrected/model/neural_preprocessing.json" --features prepared_candidates.parquet --history history.parquet --metadata meta_subset.parquet --output scores.parquet

Current outcome labels are ignored by this entry point; the input can omit them.
This is a prepared-feature scorer, not a raw-query serving API.

make_report.py regenerates the corrected Markdown report, not the legacy DOCX/PDF.
make_listing.py writes an updated Markdown inventory. make_slides.py is an optional
corrected presentation generator requiring python-pptx; its generated deck still
needs a visual review before submission. No old generator embeds draft scores.

## Interpretation

The temporal pattern is consistent with different logging windows; JD's actual
retention policy has not been verified. Content anonymization preserves term equality.
The history-conditioned CF model adapts a GMF/MLP design and is not a reproduction
of user-ID NCF. SVD uses training-only item factors with held-out history fold-in.
The neural MLP control retains the same engineered history features, isolating
the added sequence towers. Subgroup positive ranks/recall use original full slates.

This is one seed on a previously inspected test partition. Do not interpret bootstrap
intervals as training-seed variation, or offline metrics as causal business lift.
