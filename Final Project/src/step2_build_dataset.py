"""
Step 2: Build the modelling dataset (real data, reproducible sample).

The JDSearch task is a PER-QUERY RE-RANKING problem: each user record supplies one
test query, a candidate slate (~90 products) with ordinal labels
(0 no-interaction / 1 click / 2 cart / 3 purchase), and the user's preceding history.
We therefore do NOT need catalogue-wide negative sampling: the slate already contains
the negatives. Each query is a ranking "group".

We sample USERS (seed 42) for tractable training on a laptop (MPS), and write
Parquet artifacts that scale to the full data on Colab by raising N_USERS.

Outputs (artifacts/model/):
  candidates.parquet : one row per (session, candidate)  [session_id, query, cand_wid, label]
  history.parquet    : one row per (session, hist event) [session_id, pos, wid, type, qry, elapsed_days]
  sessions.parquet   : one row per session               [session_id, query, slate_size, n_hist, split]
  meta_subset.parquet: product metadata for all referenced wids
"""
import time, sys
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq

import os
SRC_USER = os.environ.get("JD_USER_TXT",
    "/Users/steff/Library/CloudStorage/OneDrive-Personal/Descargas/Steffany/Estudios/NUS/BT4222 - Mining Web Data for Business Insights/Project/JDsearch/user_behavior_data.txt")
CHUNKS = Path(os.environ.get("JD_PRODUCT_CHUNKS",
    "/Users/steff/Documents/GitHub/bt4212_finalProj/Interim Report/data/JDsearch/product_chunks"))
OUT = Path("/Users/steff/Documents/GitHub/bt4212_finalProj/Final Project/artifacts/model")
OUT.mkdir(parents=True, exist_ok=True)

N_USERS = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
MAX_HIST = 60   # keep most-recent MAX_HIST events per user (recent window matters most)
SEED = 42

def split(v):
    if v is None or str(v) == "nan" or str(v).strip() == "":
        return []
    return str(v).split("_")

def main():
    t0 = time.time()
    print(f"Sampling {N_USERS:,} users (seed {SEED}) from 173,831 ...", flush=True)
    # read only needed columns
    df = pd.read_csv(SRC_USER, sep="\t", dtype=str, low_memory=False)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(df), size=min(N_USERS, len(df)), replace=False)
    idx.sort()
    df = df.iloc[idx].reset_index(drop=True)
    print(f"  loaded+sampled ({time.time()-t0:.0f}s)", flush=True)

    cand_rows = []      # session_id, query, cand_wid, label
    hist_rows = []      # session_id, pos, wid, type, qry, elapsed_days
    sess_rows = []      # session_id, query, slate_size, n_hist
    referenced_wids = set()

    for sid, (_, r) in enumerate(df.iterrows()):
        query = r["query"] if pd.notna(r["query"]) else ""
        cw = split(r["candidate_wid_list"])
        cl = split(r["candidate_label_list"])
        if len(cw) != len(cl) or len(cw) == 0:
            continue
        labels = [int(float(x)) for x in cl]
        for w, lab in zip(cw, labels):
            cand_rows.append((sid, query, w, lab))
            referenced_wids.add(w)
        # history
        hwid = split(r["history_wid_list"]); htype = split(r["history_type_list"])
        hqry = split(r["history_qry_list"]); htime = split(r["history_time_list"])
        n = len(hwid)
        if n > 0 and len(htime) == n + 1 and len(htype) == n and len(hqry) == n:
            tarr = np.array([float(x) for x in htime])
            suffix = np.cumsum(tarr[::-1])[::-1]
            elapsed = suffix[1:] / 86400.0   # elapsed days before query, per event
            # keep most-recent MAX_HIST events (smallest elapsed)
            order = np.argsort(elapsed)       # ascending = most recent first
            keep = order[:MAX_HIST]
            keep = np.sort(keep)              # chronological
            for pos in keep:
                hist_rows.append((sid, int(pos), hwid[pos], htype[pos], hqry[pos], float(elapsed[pos])))
                referenced_wids.add(hwid[pos])
        sess_rows.append((sid, query, len(cw), n))
        if sid % 2000 == 0 and sid > 0:
            print(f"  session {sid:,}  cand_rows={len(cand_rows):,}  hist_rows={len(hist_rows):,} ({time.time()-t0:.0f}s)", flush=True)

    cand = pd.DataFrame(cand_rows, columns=["session_id","query","cand_wid","label"])
    hist = pd.DataFrame(hist_rows, columns=["session_id","pos","wid","type","qry","elapsed_days"])
    sess = pd.DataFrame(sess_rows, columns=["session_id","query","slate_size","n_hist"])

    # random split by SESSION, 70/15/15. NOTE: JDSearch gives ONE test query per user
    # with no cross-session absolute timestamps, so a global chronological split is not
    # definable across users; we therefore split by session (new-query generalization).
    rng2 = np.random.default_rng(SEED)
    perm = rng2.permutation(len(sess))
    n_tr = int(0.70*len(sess)); n_va = int(0.15*len(sess))
    split_arr = np.empty(len(sess), dtype=object)
    split_arr[perm[:n_tr]] = "train"
    split_arr[perm[n_tr:n_tr+n_va]] = "val"
    split_arr[perm[n_tr+n_va:]] = "test"
    sess["split"] = split_arr

    print(f"\nReferenced unique wids: {len(referenced_wids):,}", flush=True)
    print("Filtering product metadata chunks ...", flush=True)
    ref = referenced_wids
    metas = []
    for i, f in enumerate(sorted(CHUNKS.glob("*.parquet"))):
        t = pq.read_table(f, columns=["wid","name","brand_id","brand_name",
                                       "cate_id_1","cate_id_2","cate_id_3","cate_id_4","shop_id"])
        d = t.to_pandas()
        d = d[d["wid"].isin(ref)]
        if len(d): metas.append(d)
        print(f"  chunk {i}: matched {len(d):,} ({time.time()-t0:.0f}s)", flush=True)
    meta = pd.concat(metas, ignore_index=True).drop_duplicates("wid") if metas else pd.DataFrame()
    print(f"metadata rows for referenced wids: {len(meta):,}  "
          f"(coverage {100*len(meta)/max(len(ref),1):.1f}% of referenced)", flush=True)

    cand.to_parquet(OUT/"candidates.parquet", index=False)
    hist.to_parquet(OUT/"history.parquet", index=False)
    sess.to_parquet(OUT/"sessions.parquet", index=False)
    meta.to_parquet(OUT/"meta_subset.parquet", index=False)

    # quick summary
    print("\n=== DATASET SUMMARY ===")
    print(f"sessions: {len(sess):,}  (train {int((sess.split=='train').sum()):,} / "
          f"val {int((sess.split=='val').sum()):,} / test {int((sess.split=='test').sum()):,})")
    print(f"candidate rows: {len(cand):,}")
    print(f"label dist: {cand.label.value_counts().sort_index().to_dict()}")
    print(f"positive rate: {100*(cand.label>0).mean():.2f}%")
    print(f"history rows (capped {MAX_HIST}): {len(hist):,}")
    print(f"meta coverage of candidates: "
          f"{100*cand.cand_wid.isin(set(meta.wid)).mean():.1f}%")
    print(f"DONE step2 ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
