# Traceability

Every number reported in the slides/report is listed here with the committed file it comes from and how to
recompute it. NDCG@10 values are the TEST split unless noted. Checked against `artifacts/` and
`data/intermediate/` on 7 Nov 2026. **29 OK, 0 MISMATCH, 1 to recompute** (see the last row).

| Claim | Value | File | How to recompute (notebook / section) |
|---|---|---|---|
| User split (train/val/test) | 14,000 / 3,000 / 3,000 | `artifacts/corrected/dataset_manifest.json` (`split_counts`) | Step 2, Part B split check |
| Split fingerprint valid | `split_sha256` matches | `dataset_manifest.json` (`split_sha256`) | Step 2, Part B (prints `True`) |
| Popularity NDCG@10 | 0.3045 | `results_baselines_lgb.json` → `Popularity` | Step 4, Part B |
| SVD NDCG@10 | 0.2759 | `results_baselines_lgb.json` → `SVD(train-only fold-in)` | Step 4, Part B |
| History-CF NDCG@10 | 0.2694 | `results_neural.json` → `History-conditioned CF` | Step 5, Part B |
| LambdaMART NDCG@10 (test) | 0.3630 | `results_baselines_lgb.json` → `LambdaMART(features)` | Step 4, Part B |
| LambdaMART NDCG@10 (train) | 0.4562 | `results_baselines_lgb.json` → `..._train` | Step 4, Part B |
| LambdaMART NDCG@10 (val) | 0.3558 | `results_baselines_lgb.json` → `..._val` | Step 4, Part B |
| MLP NDCG@10 | 0.3725 | `results_neural.json` → `MLP(no towers)` | Step 5, Part B |
| DHAR NDCG@10 | 0.3713 | `results_neural.json` → `DHAR(ours)` | Step 5, Part B |
| Pooled-history NDCG@10 | 0.3738 | `results_neural.json` → `Pooled history` | Step 5, Part B |
| Ablation: full vs no temporal/action | 0.3630 vs 0.3387 (+0.0243) | `ablation.json` (`full`, `no_temporal`) | Step 6, ablation |
| Paired bootstrap DHAR-LambdaMART | +0.0083 [+0.0022, +0.0145] | `paired_bootstrap.json` → `dual_minus_lgb` | Step 6, paired bootstrap |
| Paired bootstrap DHAR-MLP | -0.0012 [-0.0034, +0.0011] | `paired_bootstrap.json` → `dual_minus_mlp` | Step 6, paired bootstrap |
| Paired bootstrap DHAR-pooled | -0.0025 [-0.0048, -0.0003] | `paired_bootstrap.json` → `dual_minus_pooled` | Step 6, paired bootstrap |
| Exposure macro Recall@10 (seen vs absent) | 0.79-0.81 vs 0.19-0.23 | `positive_subgroup_full_slate.csv` (`training_history_seen`) | Step 6, exposure diagnostic |
| Median full-slate rank: orders / carts / clicks | 5 / 10 / 18 | `rank_by_type.csv` (labels 3 / 2 / 1) | Step 6, rank-by-type |
| MLP NDCG@10 by history length | 0.381 (1-2, 63 q) / 0.392 / 0.361 / 0.375 | `segment_metrics.csv` (model `mlp`) | Step 6, history-length segments |
| Step 7: routing selected / lambda | none / 0 | `step7_corrected/coldstart_threshold_selection.csv`, `results_coldstart_diversity.json` | Step 7, Part B |
| Step 7: fallback vs full on 63 limited-history queries | 0.303 vs 0.345 | `step7_corrected/coldstart_segments.csv` (`limited(1-2)`) | Step 7 |
| Step 7: val NDCG@10 if 1-event users routed | 0.3558 -> 0.3548 | `step7_corrected/coldstart_threshold_selection.csv` | Step 7 |
| Step 7: lambda=0.05 categories / NDCG@10 | 2.04 -> 3.12 / 0.3630 -> 0.3549 | `step7_corrected/diversity_tradeoff.csv` (`lam=0.05`) | Step 7, diversity trade-off |
| Step 7: lambda=0.5 categories / NDCG@10 | 5.47 / 0.2896 | `step7_corrected/diversity_tradeoff.csv` (`lam=0.5`) | Step 7, diversity trade-off |
| Temporal: click share within one day | 85.5% | `temporal_summary.txt` | Step 1 |
| Temporal: clicks end before ~30 days | last click day-bin 13 | `retention_cutoffs.txt` (CLICK) | Step 1 |
| Temporal: carts extend to ~90 days+ | rare carts to >=365d | `retention_cutoffs.txt` (CART) | Step 1 |
| Temporal: orders up to a year | no orders >=365d | `retention_cutoffs.txt`, `temporal_type_by_daybin.csv` (ORD) | Step 1 |
| DHAR attention mass by age/type per tower | short: 0 mass >90d, cart-leaning; order: 0.58 mass >90d | `diagnostics/dhar_attention_by_age_type.csv` | Step 6, attention cell (`dhar_attention.py`) |
| Seed sensitivity (DHAR vs MLP) | DHAR 0.3731 (sd 0.0018), MLP 0.3749 (sd 0.0018); gap flips sign across seeds | `diagnostics/multiseed_summary.csv` | Step 5, multi-seed cell (`multiseed.py`) |
| **459 of 20,000 users have <=2 events** | **recompute** | not in committed files | Step 2: count users with <=2 history events across all splits (user-level, not the 63 test queries in `segment_metrics.csv`) |

The final row is a report-level claim that cannot be checked from the committed artifacts (the committed files
only expose the 63 **test** queries from <=2-event users, not the all-split user count). Recompute it in Step 2
before citing it, or drop it. See the final report's "Needs Steff" note.
