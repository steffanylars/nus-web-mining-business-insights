# BT4222 Project — Group 13 · Personalised Product Search on JDSearch

Group project for **BT4222 — Mining Web Data for Business Insights**, National University of Singapore.

- **`Interim Report/`** — interim deliverable: data acquisition, EDA notebooks, figures, product-metadata chunks.
- **`Final Project/`** — final deliverable. See **`Final Project/README.md`** for the full pipeline, results and how to run.

## Final project at a glance
Re-rank each JDSearch query's candidate slate (ordinal labels 0/1/2/3) using the query, product metadata and
user history. Key finding: a **telescoping log-retention window** (clicks ~7d, carts ~90d, orders ≥1yr) means
interaction *type is confounded with recency* — which drives horizon-aware features and our **Dual-Horizon
Attention Ranker (DHAR)**, the best model (NDCG@10 0.336) over NCF / MF / popularity / LambdaMART baselines.

Deliverables: `Final Project/report/` (report PDF+DOCX, dataset/code listing), `Final Project/slides/` (PPTX),
`Final Project/RESULTS.md`, `Final Project/AI_USAGE.md`.

> Large raw data (919 MB / 4.5 GB) and heavy intermediate artifacts are git-ignored — download JDSearch from
> https://github.com/rucliujn/JDsearch (see `Final Project/README.md`).
