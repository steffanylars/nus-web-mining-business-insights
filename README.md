# NUS BT4222 — Mining Web Data for Business Insights

Coursework for **BT4222 — Mining Web Data for Business Insights** at the **National University of Singapore**: two graded assignments plus the course final project.

## Assignments

### Assignment 1 — Fashion product recommendation
Feature engineering and a similarity-based recommender on fashion e-commerce data: customer- and item-based features, **TF-IDF** on product text, **KMeans** customer segmentation, **PCA**, and **cosine-similarity** recommendations.
📂 `Assignment-1/` · scikit-learn · pandas · (data pulled from Kaggle at runtime via `kagglehub`).

### Assignment 2 — Feature-Enhanced Neural Collaborative Filtering
A **PyTorch** recommender built on Neural Matrix Factorization / **Neural Collaborative Filtering (NCF)**, comparing an *ID-only* model vs a *feature-enhanced* model, with a **chronological train/test split** and **cold-start** handling. Also includes a critical-evaluation task auditing a flawed baseline notebook.
📂 `Assignment-2/` · PyTorch · trained model in `best_model.pt`.

> **Dataset not included.** Assignment 2 uses the **MovieLens 100K** dataset, which may not be redistributed. Download it from [GroupLens](https://grouplens.org/datasets/movielens/100k/) and place `u.data`, `u.item`, `u.user` in `Assignment-2/data/`.

## Final project
📂 `Interim Report/` — course final-project materials.

## Author
**Steffany Lara** — exchange student at NUS · Data Science & Mathematics Engineering, Tecnológico de Monterrey.
