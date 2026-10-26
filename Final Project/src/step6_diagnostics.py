"""
Step 6: Diagnostics, error analysis & ablation.
  - Model comparison bar chart (all models).
  - LambdaMART feature importance chart.
  - Segment analysis: NDCG@10 by user history length and by candidate warm/cold.
  - Rank-of-positive by interaction TYPE (click/cart/purchase) -> ties back to the
    temporal finding (purchases reflect LONG-TERM preference, clicks SHORT-TERM intent).
  - Ablation: LambdaMART WITH vs WITHOUT temporal-aware features.
"""
import json, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import lightgbm as lgb
from eval_utils import evaluate, _ranked_labels

OUT = Path("/Users/steff/Documents/GitHub/bt4212_finalProj/Final Project/artifacts/model")
RES = Path("/Users/steff/Documents/GitHub/bt4212_finalProj/Final Project/artifacts")
FIG = Path("/Users/steff/Documents/GitHub/bt4212_finalProj/Final Project/figures")

def main():
    t0=time.time()
    feat = pd.read_parquet(OUT/"features.parquet")
    sess = pd.read_parquet(OUT/"sessions.parquet")
    lgbp = pd.read_parquet(OUT/"test_pred_lgb.parquet")
    res = {}
    for jf in ["results_baselines_lgb.json","results_neural.json"]:
        p=RES/jf
        if p.exists(): res.update(json.load(open(p)))
    neural_path = OUT/"test_pred_neural.parquet"
    if neural_path.exists():
        npred = pd.read_parquet(neural_path)
        lgbp = lgbp.merge(npred[["session_id","cand_wid","dhar_score","ncf_score"]],
                          on=["session_id","cand_wid"], how="left")

    # ---------- 1. model comparison bar ----------
    order=["Random","Popularity","Content/Relevance-only","MatrixFactorisation(SVD)",
           "NCF(neural)","LambdaMART(features)","DHAR(ours)"]
    names=[m for m in order if m in res]
    metrics=["NDCG@10","MRR@200","P@1"]
    fig,ax=plt.subplots(figsize=(11,5)); x=np.arange(len(names)); w=0.26
    for i,mt in enumerate(metrics):
        ax.bar(x+i*w,[res[n][mt] for n in names],w,label=mt)
    ax.set_xticks(x+w); ax.set_xticklabels(names,rotation=25,ha="right")
    ax.set_ylabel("score (test, sessions with >=1 positive)")
    ax.set_title("Model comparison on held-out test slates")
    ax.legend(); ax.grid(axis="y",alpha=.3)
    plt.tight_layout(); fig.savefig(FIG/"fig_model_comparison.png",dpi=150); plt.close(fig)

    # ---------- 2. feature importance ----------
    imp=pd.read_csv(RES/"lgb_feature_importance.csv").head(14).iloc[::-1]
    fig,ax=plt.subplots(figsize=(8,6))
    ax.barh(imp["feature"],imp["gain"],color="#2b7bba")
    ax.set_xlabel("LightGBM gain"); ax.set_title("LambdaMART feature importance (top 14)")
    plt.tight_layout(); fig.savefig(FIG/"fig_feature_importance.png",dpi=150); plt.close(fig)

    # ---------- 3. segment analysis (LambdaMART) ----------
    lgbp=lgbp.merge(sess[["session_id","n_hist"]],on="session_id",how="left")
    def hbin(n):
        return ("cold(<3)" if n<3 else "short(3-20)" if n<=20 else "medium(21-100)" if n<=100 else "long(>100)")
    lgbp["hbin"]=lgbp["n_hist"].map(hbin)
    seg_rows=[]
    for b in ["cold(<3)","short(3-20)","medium(21-100)","long(>100)"]:
        sub=lgbp[lgbp.hbin==b]
        if sub.session_id.nunique()==0: continue
        m,n=evaluate(sub,"lgb_score")
        seg_rows.append(("history_len",b,m["NDCG@10"],m["MRR@200"],m["P@1"],n))
    # candidate warm/cold: warm if pop_wid>0 (seen in train history)
    featmap=feat.set_index(["session_id","cand_wid"])["pop_wid"].to_dict()
    lgbp["warm_item"]=lgbp.apply(lambda r:"warm" if featmap.get((r.session_id,r.cand_wid),0)>0 else "cold",axis=1)
    for b in ["warm","cold"]:
        sub=lgbp[lgbp.warm_item==b]
        # keep only sessions that still have a positive in this subset
        m,n=evaluate(sub,"lgb_score")
        seg_rows.append(("cand_warmth",b,m["NDCG@10"],m["MRR@200"],m["P@1"],n))
    seg=pd.DataFrame(seg_rows,columns=["dimension","segment","NDCG@10","MRR@200","P@1","n"])
    seg.to_csv(RES/"segment_metrics.csv",index=False)
    print("=== SEGMENT METRICS (LambdaMART) ==="); print(seg.to_string(index=False))

    # fig segment by history length
    hl=seg[seg.dimension=="history_len"]
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.bar(hl["segment"],hl["NDCG@10"],color="#2ca089")
    ax.set_ylabel("NDCG@10"); ax.set_title("LambdaMART NDCG@10 by user history length")
    for i,v in enumerate(hl["NDCG@10"]): ax.text(i,v+.005,f"{v:.3f}",ha="center")
    plt.tight_layout(); fig.savefig(FIG/"fig_segment_history.png",dpi=150); plt.close(fig)

    # ---------- 4. rank-of-positive by TYPE ----------
    # for each test session, rank candidates by lgb_score; record rank of each positive & its label
    rng=np.random.default_rng(0); recs=[]
    for sid,g in lgbp.groupby("session_id",sort=False):
        labels=g["label"].values; sc=g["lgb_score"].values.astype(float)
        if (labels>0).sum()==0: continue
        tb=rng.random(len(sc)); order=np.lexsort((tb,-sc))
        ranks=np.empty(len(order),dtype=int); ranks[order]=np.arange(1,len(order)+1)
        for lab,rk in zip(labels,ranks):
            if lab>0: recs.append((lab,rk))
    rd=pd.DataFrame(recs,columns=["label","rank"])
    rank_by_type=rd.groupby("label")["rank"].agg(["mean","median","count"])
    rank_by_type.index=["click(1)","cart(2)","purchase(3)"]
    rank_by_type.to_csv(RES/"rank_by_type.csv")
    print("\n=== MEAN RANK OF POSITIVES BY TYPE (lower=better) ===")
    print(rank_by_type.to_string())

    # ---------- 5. ablation: temporal-aware features ----------
    temporal_feats=["aff_cat_decay","aff_brand_decay","short_cat_hit","short_brand_hit",
                    "long_cat_hit","long_brand_hit","h_min_elapsed","h_n_recent7","h_n_ord","h_n_cart","h_n_click"]
    drop={"session_id","cand_wid","label","split"}
    allcols=[c for c in feat.columns if c not in drop]
    def groups(df): return df.groupby("session_id",sort=False).size().values
    def train_eval(cols,tag):
        tr=feat[feat.split=="train"].sort_values("session_id")
        va=feat[feat.split=="val"].sort_values("session_id")
        te=feat[feat.split=="test"].sort_values("session_id")
        dtr=lgb.Dataset(tr[cols],label=tr["label"],group=groups(tr))
        dva=lgb.Dataset(va[cols],label=va["label"],group=groups(va),reference=dtr)
        params=dict(objective="lambdarank",metric="ndcg",ndcg_eval_at=[10],learning_rate=0.05,
                    num_leaves=63,min_data_in_leaf=200,label_gain=[0,1,3,7],verbosity=-1,seed=42)
        b=lgb.train(params,dtr,num_boost_round=400,valid_sets=[dva],
                    callbacks=[lgb.early_stopping(50,verbose=False)])
        te=te.copy(); te["s"]=b.predict(te[cols])
        m,_=evaluate(te,"s"); print(f"  {tag:30s} NDCG@10 {m['NDCG@10']:.4f} MRR@200 {m['MRR@200']:.4f} P@1 {m['P@1']:.4f}")
        return m
    print("\n=== ABLATION: temporal-aware features ===")
    full=train_eval(allcols,"FULL (all features)")
    notemp=train_eval([c for c in allcols if c not in temporal_feats],"WITHOUT temporal features")
    abl={"full":full,"no_temporal":notemp,
         "delta_NDCG@10":full["NDCG@10"]-notemp["NDCG@10"]}
    json.dump(abl,open(RES/"ablation.json","w"),indent=2)
    print(f"  --> temporal features add NDCG@10 +{abl['delta_NDCG@10']:.4f}")
    print(f"\nDONE step6 ({time.time()-t0:.0f}s)")

if __name__=="__main__":
    main()
