"""Features from complete preceding histories; short clicks <=7d / carts <=90d, long orders.

Memory-efficient build (fits ~8 GB, e.g. free Colab): metadata is read with only the
columns the features need, name/category/brand tokens are built for candidate items only,
category and brand names (low cardinality) share one tokenised set per distinct string,
large intermediates are freed as soon as they are done, and candidates are processed in
chunks. None of this changes a feature value; the output is byte-for-byte the same table.
"""
import gc
import time
import numpy as np, pandas as pd

from config import OUT, RES, prepare, save_json
from data_utils import tok, short_mask

# Only these product columns feed the features (Steps 3/5/7 never read the rest).
META_COLS = ["wid", "cate_id_3", "brand_id", "name", "cate_name_3", "brand_name"]
CAND_CHUNK = 300000  # candidate rows per block; bounds the transient row buffer

def intern_tokens(series):
    """Tokenise each distinct string once and share the resulting set.

    Category and brand names repeat across millions of products (a few thousand
    distinct values), so one set per distinct string, referenced by every wid that
    carries it, gives identical lookups at a fraction of the memory.
    """
    cache = {}
    out = {}
    for w, v in series.items():
        s = cache.get(v)
        if s is None:
            s = tok(v); cache[v] = s
        out[w] = s
    return out

def main():
    prepare()
    t0=time.time()
    cand = pd.read_parquet(OUT/"candidates.parquet")
    hist = pd.read_parquet(OUT/"history.parquet")
    sess = pd.read_parquet(OUT/"sessions.parquet")
    meta = pd.read_parquet(OUT/"meta_subset.parquet", columns=META_COLS)
    print(f"loaded: cand {len(cand):,}, hist {len(hist):,}, sess {len(sess):,}, meta {len(meta):,} ({time.time()-t0:.0f}s)")

    meta = meta.set_index("wid")
    # ---- maps wid -> category / brand (every referenced item needs these) ----
    cate3 = meta["cate_id_3"].to_dict()
    brand = meta["brand_id"].to_dict()
    # ---- name/category/brand tokens for candidate items only (content features are candidate-side) ----
    cand_wids = set(cand["cand_wid"].unique())
    mc = meta[meta.index.isin(cand_wids)]
    name_tok = {w: tok(n) for w,n in mc["name"].items()}
    cname3 = intern_tokens(mc["cate_name_3"]) if "cate_name_3" in meta else {}
    bname  = intern_tokens(mc["brand_name"])  if "brand_name" in meta else {}
    del meta, mc; gc.collect()

    train_sids = set(sess.loc[sess.split=="train","session_id"])

    # ---- POPULARITY PRIORS from TRAIN histories only (no leakage) ----
    hist_tr = hist[hist.session_id.isin(train_sids)]
    wid_pop = hist_tr["wid"].value_counts().to_dict()
    hist_tr_c3 = hist_tr["wid"].map(cate3)
    cate3_pop = hist_tr_c3.value_counts().to_dict()
    hist_tr_b = hist_tr["wid"].map(brand)
    brand_pop = hist_tr_b.value_counts().to_dict()
    del hist_tr, hist_tr_c3, hist_tr_b; gc.collect()
    print(f"priors built ({time.time()-t0:.0f}s)")

    # ---- Per-session history profiles (temporal-aware) ----
    # keep only the columns the profiles read; elapsed_days stays float64 (feeds exp/min exactly)
    hist = hist[["session_id","wid","type","qry","elapsed_days"]].copy()
    hist["c3"] = hist["wid"].map(cate3)
    hist["b"]  = hist["wid"].map(brand)
    hist["is_click"] = (hist["type"]=="CLICK")
    hist["is_cart"]  = (hist["type"]=="CART")
    hist["is_ord"]   = (hist["type"]=="ORD")
    hist["recent7"]  = (hist["elapsed_days"]<=7)
    hist["intent"] = short_mask(hist)

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
    short_c3 = g.apply(lambda d: set(d.loc[d.intent,"c3"].dropna()), include_groups=False).to_dict()
    short_b  = g.apply(lambda d: set(d.loc[d.intent,"b"].dropna()), include_groups=False).to_dict()
    # long-term preference categories/brands from ORDERS (any age)
    long_c3  = g.apply(lambda d: set(d.loc[d.is_ord,"c3"].dropna()), include_groups=False).to_dict()
    long_b   = g.apply(lambda d: set(d.loc[d.is_ord,"b"].dropna()), include_groups=False).to_dict()
    # All exact item IDs from preceding history
    seen_wid = g["wid"].apply(set).to_dict()
    # recency-decayed category affinity (tau=30d) and brand affinity
    aff_c3 = {}; aff_b = {}
    for sid, d in g:
        wv = np.exp(-d["elapsed_days"].values/30.0)
        c3v = d["c3"].values; bv = d["b"].values
        ac={}; ab={}
        for c,w in zip(c3v,wv):
            if pd.notna(c): ac[c]=ac.get(c,0)+w
        for b,w in zip(bv,wv):
            if pd.notna(b): ab[b]=ab.get(b,0)+w
        aff_c3[sid]=ac; aff_b[sid]=ab
    # All preceding history-query tokens per session
    sess_qtok = g.apply(lambda d: set().union(*[tok(q) for q in d["qry"] if q!="-1"]) if (d["qry"]!="-1").any() else set(), include_groups=False).to_dict()
    del hist, g; gc.collect()
    # the candidate loop below only looks up candidate items, so shrink the id/popularity
    # maps to candidate wids and free the full-catalogue versions (value-identical lookups)
    cate3 = {w: cate3[w] for w in cand_wids if w in cate3}
    brand = {w: brand[w] for w in cand_wids if w in brand}
    wid_pop = {w: wid_pop[w] for w in cand_wids if w in wid_pop}
    gc.collect()
    print(f"session profiles built ({time.time()-t0:.0f}s)")

    # ---- build feature rows per candidate, in chunks ----
    cand = cand.merge(sess[["session_id","split","slate_size"]], on="session_id", how="left")
    cw = cand["cand_wid"].values; qs = cand["query"].values; sids = cand["session_id"].values
    lab_values = cand["label"].values; split_values = cand["split"].values
    slate_values = cand["slate_size"].values
    query_tokens = {q: tok(q) for q in cand["query"].unique()}
    del cand; gc.collect()
    cols = ["session_id","cand_wid","label","split",
            "query_len","slate_size","has_meta",
            "ov_name","ov_cat","ov_brand","jacc_name",
            "pop_wid","pop_cate3","pop_brand",
            "short_cat_hit","short_brand_hit","long_cat_hit","long_brand_hit",
            "aff_cat_decay","aff_brand_decay","seen_before","q_hist_overlap"]
    n_cand = len(cw); blocks = []
    for start in range(0, n_cand, CAND_CHUNK):
        stop = min(start + CAND_CHUNK, n_cand)
        rows = []
        for i in range(start, stop):
            w = cw[i]; sid = sids[i]
            qt = query_tokens[qs[i]]
            c3 = cate3.get(w); b = brand.get(w)
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
            rows.append((sid, w, lab_values[i], split_values[i],
                         len(qt), slate_values[i], has_meta,
                         ov_name, ov_cat, ov_brand, jacc_name,
                         pop_w, pop_c3, pop_b,
                         short_cat_hit, short_brand_hit, long_cat_hit, long_brand_hit,
                         aff_cat, aff_brand, seen, q_hist_overlap))
        blocks.append(pd.DataFrame(rows, columns=cols))
        del rows
        print(f"  feat {stop:,}/{n_cand:,} ({time.time()-t0:.0f}s)", flush=True)
    # every lookup is consumed; free them before the concat/merge/astype tail
    del cate3, brand, name_tok, cname3, bname, wid_pop, cate3_pop, brand_pop
    del short_c3, short_b, long_c3, long_b, seen_wid, aff_c3, aff_b, sess_qtok, query_tokens
    del cw, qs, sids, lab_values, split_values, slate_values; gc.collect()
    feat = pd.concat(blocks, ignore_index=True) if len(blocks) > 1 else blocks[0]
    del blocks; gc.collect()

    feat = feat.merge(sess_feat, on="session_id", how="left")
    hist_cols = ["h_len","h_n_click","h_n_cart","h_n_ord","h_n_recent7","h_n_distinct_c3","h_min_elapsed","h_queryless"]
    for c in hist_cols: feat[c] = feat[c].fillna(0)
    numeric = [c for c in feat if c not in {"session_id","cand_wid","split","label"}]
    feat[numeric] = feat[numeric].astype("float32")
    if not np.isfinite(feat[numeric].to_numpy()).all():
        raise ValueError("Nonfinite numeric features")
    feat.to_parquet(OUT/"features.parquet", index=False)
    save_json(RES/"feature_summary.json", {
        "rows": len(feat), "numeric_features": numeric,
        "means_by_label": {str(k): {c:float(v) for c,v in row.items()}
                          for k,row in feat.groupby("label")[numeric].mean().iterrows()},
        "nonzero_content_pairs": {c:int((feat[c]>0).sum()) for c in ["ov_name","ov_cat","ov_brand"]},
        "short_policy": "CLICK <=7d or CART <=90d; all preceding orders in long profile",
        "h_min_elapsed_definition": "age of most recent user event, not candidate-specific recency"
    })

    print("\n=== FEATURES SUMMARY ===")
    print(f"rows: {len(feat):,}  cols: {feat.shape[1]}")
    print(f"splits: {feat.split.value_counts().to_dict()}")
    print("mean feature values by label (sample):")
    show = ["ov_name","jacc_name","pop_wid","short_cat_hit","long_cat_hit","aff_cat_decay","seen_before","has_meta"]
    print(feat.groupby("label")[show].mean().to_string())
    print(f"DONE step3 ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
