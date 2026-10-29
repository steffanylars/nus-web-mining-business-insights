# Corrected measured results

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

| Architecture | Validation NDCG@10 | Test NDCG@10 |
|---|---|---|
| MLP(no towers) | 0.3676 | 0.3725 |
| Pooled history | 0.3679 | 0.3738 |
| Short only | 0.3692 | 0.3713 |
| Order only | 0.3678 | 0.3731 |
| DHAR(ours) | 0.3689 | 0.3713 |

DHAR differs from the same-feature MLP by -0.0012 NDCG@10 (paired query bootstrap 95% interval [-0.0034, +0.0011]). This run does not establish an advantage for dual attention beyond the engineered features.

Sources: results_baselines_lgb.json, results_neural.json, ablation.json and paired_bootstrap.json. See REPORT.md for protocol and limitations.
