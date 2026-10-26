# BT4222 Final Project — Group 13
## Personalised Product Search on JDSearch: A Dual-Horizon Re-Ranker Driven by Log-Retention Dynamics

Re-rank each JDSearch query's candidate slate so that products the shopper later
clicks / carts / purchases rise to the top, using the current query, product
metadata and the user's multi-behaviour history. Labels are ordinal
(0 none / 1 click / 2 cart / 3 purchase); metrics are NDCG@10, MRR@200, P@1.

### Headline result
Our **Dual-Horizon Attention Ranker (DHAR)** reaches **NDCG@10 0.335 / MRR@200 0.350 / P@1 0.225**,
beating LambdaMART (0.313), SVD/NMF (0.274), popularity (0.270) and ID-based NCF (0.086 ≈ random).

### The core finding (instructor's prompt)
Tracing history back in time (seconds → elapsed days before the query) reveals a
**telescoping log-retention window**: clicks are kept ~7 days, carts ~90 days, orders ≥1 year.
So interaction **type is confounded with recency** — recent clicks/carts encode short-term
*intent*, old orders encode long-term *preference*. This insight drives every feature and the model.

## Pipeline (`src/`) — run in order
| Step | Script | Purpose | Key output |
|------|--------|---------|------------|
| 1 | `step1_preprocess_temporal.py` | Parse full data; temporal retention analysis | `artifacts/temporal_*`, Figs 1 |
| 1b | `step1b_retention_cutoffs.py` | Pin exact 7d/90d cut-offs | `artifacts/retention_cutoffs.txt`, Fig 2 |
| 2 | `step2_build_dataset.py [N_USERS]` | Build modelling dataset (sampled) | `artifacts/model/*.parquet` |
| 3 | `step3_features.py` | Horizon-aware feature engineering | `artifacts/model/features.parquet` |
| 4 | `step4_baselines_lgbm.py` | Random/Popularity/Content/MF + LambdaMART | `results_baselines_lgb.json` |
| 5 | `step5_neural.py` | DHAR (ours) + NCF baseline (PyTorch/MPS) | `results_neural.json`, `dhar.pt` |
| 6 | `step6_diagnostics.py` | Segments, rank-by-type, ablation | `segment_metrics.csv`, `ablation.json` |
| — | `make_report.py` / `make_slides.py` | Build report (.docx/.pdf) and slides (.pptx) | `report/`, `slides/` |

**Scaling:** the whole pipeline ran on a 20,000-user sample on a laptop (Apple MPS).
Raise `N_USERS` in step 2 (up to 173,831) to reproduce on the full data in Colab/GPU — one knob.

## Deliverables
- `report/BT4222_Group13_Final_Report.pdf` (and `.docx`)
- `slides/BT4222_Group13_Final_Presentation.pptx`
- `RESULTS.md` — preserved performance outputs of every step
- `figures/` — all report/slide figures
- `artifacts/` — result tables (JSON/CSV) + trained models (`lgb_model.txt`, `dhar.pt`)

## Data
JDSearch (Liu et al., SIGIR 2023): https://github.com/rucliujn/JDsearch — download the full
`user_behavior_data.txt` (919 MB) and `product_meta_data.txt` (4.5 GB) from the JD Cloud link
in that repo. Large raw/intermediate files are git-ignored; set the paths at the top of each
`src/step*.py`. Product metadata is pre-chunked to Parquet under `Interim Report/data/JDsearch/product_chunks/`.

## Environment
Python 3.12 · pandas · numpy · scikit-learn · lightgbm 4.7 · torch 2.7 (MPS) · python-docx · python-pptx.
