"""
Step 4: Baselines + feature-based LambdaMART (advanced model A).

Baselines (prior-project style, for gap diagnosis):
  - Random               : sanity floor
  - Popularity           : rank by train-history popularity prior (non-personalised)
  - Content / Relevance  : rank by query<->candidate token overlap (relevance-only)
  - Matrix Factorisation : SVD on user x historical-item implicit matrix (NCF/NMF family);
                           exposes the COLD-ITEM ceiling (80% of catalogue never in history)
Advanced A:
  - LambdaMART (LightGBM): listwise learning-to-rank on the temporal-aware features.

Evaluation: NDCG@10, MRR@200, P@1, HR@10 on the held-out TEST sessions.
"""
import time, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
from eval_utils import evaluate

OUT = Path("/Users/steff/Documents/GitHub/bt4212_finalProj/Final Project/artifacts/model")
RES = Path("/Users/steff/Documents/GitHub/bt4212_finalProj/Final Project/artifacts")

def main():
    t0 = time.time()
    feat = pd.read_parquet(OUT/"features.parquet")
    hist = pd.read_parquet(OUT/"history.parquet")
    rng = np.random.default_rng(42)
    feat["rand"] = rng.random(len(feat))
    feat["content_score"] = feat["ov_name"] + feat["ov_cat"] + 0.5*feat["ov_brand"] + feat["jacc_name"]
    test = feat[feat.split=="test"].copy()
    results = {}

    def report(name, score_col, df=test):
        m, n = evaluate(df, score_col)
        results[name] = m
        print(f"{name:28s} | NDCG@10 {m['NDCG@10']:.4f} | MRR@200 {m['MRR@200']:.4f} | "
              f"P@1 {m['P@1']:.4f} | HR@10 {m['HR@10']:.4f}  (n={n})")

    print("=== BASELINES (test) ===")
    report("Random", "rand")
    report("Popularity", "pop_wid")
    report("Content/Relevance-only", "content_score")

    # ---- Matrix Factorisation (SVD) on user x historical-item ----
    print(f"\nBuilding user x item implicit matrix from history ... ({time.time()-t0:.0f}s)")
    item_ids = {w:i for i,w in enumerate(hist["wid"].unique())}
    hist2 = hist.assign(ii=hist["wid"].map(item_ids))
    rows = hist2["session_id"].values; cols = hist2["ii"].values
    data = np.ones(len(hist2))
    n_sess = feat["session_id"].max()+1
    M = csr_matrix((data,(rows,cols)), shape=(n_sess, len(item_ids)))
    k = 64
    svd = TruncatedSVD(n_components=k, random_state=42)
    U = svd.fit_transform(M)            # session latent
    V = svd.components_.T               # item latent
    print(f"  SVD done: U{U.shape} V{V.shape} ({time.time()-t0:.0f}s)")
    cand_ii = test["cand_wid"].map(item_ids)
    warm = cand_ii.notna().values
    mf_score = np.zeros(len(test))
    ci = cand_ii.fillna(-1).astype(int).values
    sid = test["session_id"].values
    uv = U[sid]
    for j in range(len(test)):
        if warm[j]:
            mf_score[j] = uv[j] @ V[ci[j]]
    test["mf_score"] = mf_score
    print(f"  MF warm-candidate coverage on test: {100*warm.mean():.1f}%")
    report("MatrixFactorisation(SVD)", "mf_score")

    # ---- LambdaMART ----
    print(f"\nTraining LambdaMART ... ({time.time()-t0:.0f}s)")
    drop = {"session_id","cand_wid","label","split","rand","content_score"}
    featu_cols = [c for c in feat.columns if c not in drop]
    def groups(df): return df.groupby("session_id", sort=False).size().values
    tr = feat[feat.split=="train"].sort_values("session_id")
    va = feat[feat.split=="val"].sort_values("session_id")
    te = test.sort_values("session_id")
    dtr = lgb.Dataset(tr[featu_cols], label=tr["label"], group=groups(tr))
    dva = lgb.Dataset(va[featu_cols], label=va["label"], group=groups(va), reference=dtr)
    params = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[10],
                  learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
                  label_gain=[0,1,3,7], verbosity=-1, seed=42)
    booster = lgb.train(params, dtr, num_boost_round=600, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    te = te.copy(); te["lgb_score"] = booster.predict(te[featu_cols])
    report("LambdaMART(features)", "lgb_score", te)

    # TRAIN performance (overfitting check) for a baseline + the advanced model
    trn = tr.copy(); trn["pop_wid_s"] = trn["pop_wid"]; trn["lgb_score"] = booster.predict(trn[featu_cols])
    mtr_pop,_ = evaluate(trn, "pop_wid_s"); mtr_lgb,_ = evaluate(trn, "lgb_score")
    results["Popularity_train"] = mtr_pop; results["LambdaMART(features)_train"] = mtr_lgb
    print(f"\nTRAIN perf  | Popularity NDCG@10 {mtr_pop['NDCG@10']:.4f} | "
          f"LambdaMART NDCG@10 {mtr_lgb['NDCG@10']:.4f} (test {results['LambdaMART(features)']['NDCG@10']:.4f})")

    # feature importance
    imp = pd.DataFrame({"feature":featu_cols,
                        "gain":booster.feature_importance("gain")}).sort_values("gain",ascending=False)
    imp.to_csv(RES/"lgb_feature_importance.csv", index=False)
    print("\nTop features by gain:")
    print(imp.head(12).to_string(index=False))

    # save predictions for diagnostics (step6)
    te[["session_id","cand_wid","label","lgb_score"]].to_parquet(OUT/"test_pred_lgb.parquet", index=False)
    booster.save_model(str(OUT/"lgb_model.txt"))
    with open(RES/"results_baselines_lgb.json","w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDONE step4 ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
