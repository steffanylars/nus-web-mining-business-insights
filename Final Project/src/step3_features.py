"""
Step 3: Feature engineering (temporal-aware), driven by the Step-1 finding.

KEY INSIGHT (Step 1): interaction TYPE is confounded with RECENCY because of a
telescoping log-retention window (clicks kept 7d, carts 90d, orders >=1yr).
=> We must NOT treat type as pure intent. Instead we build features by HORIZON
   and by ROLE of each field:
     - SHORT-TERM INTENT  : categories/brands/queries from RECENT events (<=7d clicks, <=90d carts)
     - LONG-TERM PREFERENCE: categories/brands from ORDER history (any age)
     - CONTENT RELEVANCE   : query<->candidate token overlap (name/brand/category)
     - POPULARITY PRIOR    : candidate/brand/category frequency (TRAIN histories only -> no leakage)
     - EXACT RE-ENGAGEMENT : has the user touched this exact product / its category before, and how recently

Output: artifacts/model/features.parquet  (one row per session-candidate)
"""
import time
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path("/Users/steff/Documents/GitHub/bt4212_finalProj/Final Project/artifacts/model")

def tok(s):
    if s is None or (isinstance(s,float)) or str(s)=="nan" or str(s).strip()=="":
        return set()
    return set(str(s).split("^X")) if "^X" in str(s) else set(str(s).split("_"))

def main():
    t0=time.time()
    cand = pd.read_parquet(OUT/"candidates.parquet")
    hist = pd.read_parquet(OUT/"history.parquet")
    sess = pd.read_parquet(OUT/"sessions.parquet")
    meta = pd.read_parquet(OUT/"meta_subset.parquet")
    print(f"loaded: cand {len(cand):,}, hist {len(hist):,}, sess {len(sess):,}, meta {len(meta):,} ({time.time()-t0:.0f}s)")

    meta = meta.set_index("wid")
    # ---- maps wid -> category / brand / name-tokens ----
    cate3 = meta["cate_id_3"].to_dict()
    cate1 = meta["cate_id_1"].to_dict()
    brand = meta["brand_id"].to_dict()
    name_tok = {w: tok(n) for w,n in meta["name"].items()}
    cname3 = {w: tok(n) for w,n in meta["cate_name_3"].items()} if "cate_name_3" in meta else {}
    bname  = {w: tok(n) for w,n in meta["brand_name"].items()} if "brand_name" in meta else {}

    train_sids = set(sess.loc[sess.split=="train","session_id"])

    # ---- POPULARITY PRIORS from TRAIN histories only (no leakage) ----
    hist_tr = hist[hist.session_id.isin(train_sids)]
    wid_pop = hist_tr["wid"].value_counts().to_dict()
    hist_tr_c3 = hist_tr["wid"].map(cate3)
    cate3_pop = hist_tr_c3.value_counts().to_dict()
    hist_tr_b = hist_tr["wid"].map(brand)
    brand_pop = hist_tr_b.value_counts().to_dict()
    print(f"priors built ({time.time()-t0:.0f}s)")

    # ---- Per-session history profiles (temporal-aware) ----
    # attach category/brand to history
    hist = hist.copy()
    hist["c3"] = hist["wid"].map(cate3)
    hist["c1"] = hist["wid"].map(cate1)
    hist["b"]  = hist["wid"].map(brand)
    hist["is_click"] = (hist["type"]=="CLICK")
    hist["is_cart"]  = (hist["type"]=="CART")
    hist["is_ord"]   = (hist["type"]=="ORD")
    hist["recent7"]  = (hist["elapsed_days"]<=7)
    hist["recent90"] = (hist["elapsed_days"]<=90)

    # per-session aggregates
    g = hist.groupby("session_id")
    sess_feat = pd.DataFrame({
        "h_len": g.size(),
        "h_n_click": g["is_click"].sum(),
        "h_n_cart":  g["is_cart"].sum(),
        "h_n_ord":   g["is_ord"].sum(),
        "h_n_recent7": g["recent7"].sum(),
        "h_n_distinct_c3": g["c3"].nunique(),
        "h_min_elapsed": g["elapsed_days"].min(),
        "h_queryless": g.apply(lambda d: (d["qry"]=="-1").mean(), include_groups=False),
    }).reset_index()

    # per-session horizon category/brand sets + recency-weighted affinity dicts
    # short-term intent categories (recent clicks+carts, <=7d emphasises clicks)
    short_c3 = g.apply(lambda d: set(d.loc[d.recent7,"c3"].dropna()), include_groups=False).to_dict()
    short_b  = g.apply(lambda d: set(d.loc[d.recent7,"b"].dropna()), include_groups=False).to_dict()
    # long-term preference categories/brands from ORDERS (any age)
    long_c3  = g.apply(lambda d: set(d.loc[d.is_ord,"c3"].dropna()), include_groups=False).to_dict()
    long_b   = g.apply(lambda d: set(d.loc[d.is_ord,"b"].dropna()), include_groups=False).to_dict()
    # all interacted exact wids and their min elapsed
    seen_wid = g["wid"].apply(set).to_dict()
    # recency-decayed category affinity (tau=30d) and brand affinity
    def decay_aff(d):
        w = np.exp(-d["elapsed_days"].values/30.0)
        return d.assign(_w=w)
    aff_c3 = {}; aff_b = {}
    for sid, d in g:
        wv = np.exp(-d["elapsed_days"].values/30.0)
        c3v = d["c3"].values; bv = d["b"].values
        ac={}; ab={}
        for c,w in zip(c3v,wv):
            if c==c or c is not None: ac[c]=ac.get(c,0)+w
        for b,w in zip(bv,wv):
            if b==b or b is not None: ab[b]=ab.get(b,0)+w
        aff_c3[sid]=ac; aff_b[sid]=ab
    # recent-history query tokens per session
    sess_qtok = g.apply(lambda d: set().union(*[tok(q) for q in d["qry"] if q!="-1"]) if (d["qry"]!="-1").any() else set(), include_groups=False).to_dict()
    print(f"session profiles built ({time.time()-t0:.0f}s)")

    # ---- build feature rows per candidate ----
    cand = cand.merge(sess[["session_id","split","slate_size"]], on="session_id", how="left")
    rows = []
    cw = cand["cand_wid"].values; qs = cand["query"].values; sids = cand["session_id"].values
    for i in range(len(cand)):
        w = cw[i]; sid = sids[i]
        qt = tok(qs[i])
        c3 = cate3.get(w); c1 = cate1.get(w); b = brand.get(w)
        has_meta = 1 if w in name_tok else 0
        # content relevance
        nt = name_tok.get(w,set()); ct = cname3.get(w,set()); bt = bname.get(w,set())
        ov_name = len(qt & nt); ov_cat = len(qt & ct); ov_brand = len(qt & bt)
        jacc_name = ov_name/max(len(qt|nt),1)
        # popularity priors (log)
        pop_w = np.log1p(wid_pop.get(w,0))
        pop_c3 = np.log1p(cate3_pop.get(c3,0))
        pop_b = np.log1p(brand_pop.get(b,0))
        # short/long affinity
        sc = short_c3.get(sid,set()); sb = short_b.get(sid,set())
        lc = long_c3.get(sid,set()); lb = long_b.get(sid,set())
        short_cat_hit = 1 if (c3 is not None and c3 in sc) else 0
        short_brand_hit = 1 if (b is not None and b in sb) else 0
        long_cat_hit  = 1 if (c3 is not None and c3 in lc) else 0
        long_brand_hit = 1 if (b is not None and b in lb) else 0
        aff_cat = aff_c3.get(sid,{}).get(c3,0.0)
        aff_brand = aff_b.get(sid,{}).get(b,0.0)
        # exact re-engagement
        seen = 1 if w in seen_wid.get(sid,set()) else 0
        # query overlap with user's past queries
        q_hist_overlap = len(qt & sess_qtok.get(sid,set()))
        rows.append((sid, w, cand["label"].values[i], cand["split"].values[i],
                     len(qt), cand["slate_size"].values[i], has_meta,
                     ov_name, ov_cat, ov_brand, jacc_name,
                     pop_w, pop_c3, pop_b,
                     short_cat_hit, short_brand_hit, long_cat_hit, long_brand_hit,
                     aff_cat, aff_brand, seen, q_hist_overlap))
        if i % 200000 == 0 and i>0:
            print(f"  feat {i:,}/{len(cand):,} ({time.time()-t0:.0f}s)", flush=True)

    cols = ["session_id","cand_wid","label","split",
            "query_len","slate_size","has_meta",
            "ov_name","ov_cat","ov_brand","jacc_name",
            "pop_wid","pop_cate3","pop_brand",
            "short_cat_hit","short_brand_hit","long_cat_hit","long_brand_hit",
            "aff_cat_decay","aff_brand_decay","seen_before","q_hist_overlap"]
    feat = pd.DataFrame(rows, columns=cols)
    feat = feat.merge(sess_feat, on="session_id", how="left")
    hist_cols = ["h_len","h_n_click","h_n_cart","h_n_ord","h_n_recent7","h_n_distinct_c3","h_min_elapsed","h_queryless"]
    for c in hist_cols: feat[c] = feat[c].fillna(0)
    feat.to_parquet(OUT/"features.parquet", index=False)

    print("\n=== FEATURES SUMMARY ===")
    print(f"rows: {len(feat):,}  cols: {feat.shape[1]}")
    print(f"splits: {feat.split.value_counts().to_dict()}")
    print("mean feature values by label (sample):")
    show = ["ov_name","jacc_name","pop_wid","short_cat_hit","long_cat_hit","aff_cat_decay","seen_before","has_meta"]
    print(feat.groupby("label")[show].mean().to_string())
    print(f"DONE step3 ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
