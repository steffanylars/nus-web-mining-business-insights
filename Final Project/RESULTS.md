# Preserved Performance Outputs (all steps, 20,000-user sample, seed 42)

## Step 1 — Temporal retention analysis (FULL data, 173,831 users, 26,667,260 events)

Interaction-type composition by elapsed time before the test query:

| Elapsed | CLICK | CART | ORD | n events |
|---|---|---|---|---|
| ≤1 min | 84.41% | 13.30% | 2.27% | 76,058 |
| 1min–1h | 85.73% | 13.03% | 1.22% | 650,760 |
| 1h–1d | 85.51% | 12.69% | 1.77% | 1,515,756 |
| 1–7 d | 83.99% | 13.79% | 2.19% | 7,596,843 |
| 7–30 d | 0.00% | 85.48% | 14.32% | 3,861,484 |
| 30–90 d | 0.00% | 84.18% | 15.64% | 8,183,797 |
| 90–180 d | 0.00% | 0.07% | 99.93% | 1,825,329 |
| 180–365 d | 0.00% | 0.00% | 100.00% | 2,957,232 |

**Exact retention cut-offs (1-day resolution):** CLICK day6=520,760 → day7=**5** (≈7-day window);
CART day85=100,902 → day90=**365** → day95=56 (≈90-day window); ORD persists ≥1 year.
Totals: CLICK 8,298,894 · CART 11,526,504 · ORD ≈6.82M · FLW ≈24k.

## Step 2 — Modelling dataset
20,000 sessions (train 14,000 / val 3,000 / test 3,000) · 1,781,978 candidate rows ·
label dist {0: 1,745,400 · 1: 25,480 · 2: 8,709 · 3: 2,389} · positive rate **2.05%** ·
history rows (cap 60): 907,554 · candidate metadata coverage 99.5% · MF warm-candidate coverage 14.5%.

## Step 4 — Baselines + LambdaMART (test, n=2,965 sessions with ≥1 positive)

| Model | NDCG@10 | MRR@200 | P@1 | HR@10 |
|---|---|---|---|---|
| Random | 0.083 | 0.102 | 0.033 | 0.227 |
| Popularity | 0.270 | 0.283 | 0.167 | 0.518 |
| Content / relevance-only | 0.087 | 0.108 | 0.039 | 0.232 |
| Matrix Factorisation (SVD) | 0.274 | 0.289 | 0.170 | 0.518 |
| **LambdaMART (features)** | **0.313** | **0.325** | **0.204** | **0.579** |

Top features by gain: pop_wid, seen_before, pop_brand, slate_size, h_min_elapsed, pop_cate3,
aff_cat_decay, h_queryless, aff_brand_decay, h_n_ord, h_n_cart, h_n_distinct_c3.

**Note on evaluation integrity:** before fixing position leakage (random tie-breaking), the
constant-score "content-only" ranker spuriously scored NDCG 0.965 because the planted positive is
first in 98.8% of slates. After the fix it is 0.087 (≈random) — the honest value.

## Step 5 — Neural models (PyTorch / Apple MPS)

| Model | NDCG@10 | MRR@200 | P@1 | HR@10 |
|---|---|---|---|---|
| NCF (ID-based CF) | 0.086 | 0.104 | 0.033 | 0.229 |
| **DHAR (ours)** | **0.336** | **0.351** | **0.229** | **0.603** |

NCF ≈ random → confirms the CF structural gap (new queries + 80.5% cold items).
Feature standardisation uses TRAIN statistics only (no transductive leakage).

**Train vs test (overfitting check):** Popularity 0.292→0.270 · LambdaMART 0.394→0.313 (gap 0.081) ·
DHAR 0.349→0.336 (gap 0.013). DHAR generalises best — smallest train–test gap and highest test score.

## Step 6 — Diagnostics

**Segment NDCG@10 (LambdaMART):** warm candidate 0.586 vs **cold candidate 0.180** (cold-product ceiling);
cold users 0.323 · short 0.330 · medium 0.304 · long-history 0.311 (no degradation with history length).

**Mean/median rank of positives by type (lower=better):**
click 39.8 / **26** · cart 32.0 / **15** · purchase 24.7 / **7**.
→ The model ranks eventual purchases highest, validating that purchases align with the stable
long-term order signal while clicks are noisier short-term intent.

**Ablation (controlled, same LambdaMART config):** FULL NDCG@10 0.295 vs WITHOUT temporal-aware
features 0.282 → temporal features add **+0.013 NDCG@10**.
