"""Shared ranking-evaluation utilities. Groups = sessions (one query each).

IMPORTANT (position-leakage guard): JDSearch candidate lists are NOT in display
order, and the FIRST listed candidate is positive in 98.8% of slates. A stable
argsort would leak this through TIE-BREAKING (e.g. constant/zero scores keep the
original order and thus rank the planted positive first). We therefore break ties
with a fixed-seed RANDOM key (np.lexsort), exactly matching the interim report's
rule "candidate order will be shuffled before evaluation and not used as a feature".
"""
import numpy as np, pandas as pd

def _ranked_labels(labels, scores, tiebreak):
    # primary: high score first; secondary: random tiebreak
    order = np.lexsort((tiebreak, -scores))
    return np.asarray(labels)[order]

def _dcg(rels, k):
    rels = np.asarray(rels, dtype=float)[:k]
    if rels.size == 0: return 0.0
    disc = 1.0/np.log2(np.arange(2, rels.size+2))
    return float(((2**rels - 1) * disc).sum())

def _metrics_from_ranked(ranked, ks):
    res = {}
    ideal = np.sort(ranked)[::-1]
    for k in ks:
        idcg = _dcg(ideal, k)
        res[f"NDCG@{k}"] = _dcg(ranked, k)/idcg if idcg > 0 else 0.0
    pos = (ranked > 0).astype(int)
    nz = np.nonzero(pos[:200])[0]
    res["MRR@200"] = 1.0/(nz[0]+1) if nz.size else 0.0
    res["P@1"] = 1.0 if ranked[0] > 0 else 0.0
    res["HR@10"] = 1.0 if (ranked[:10] > 0).any() else 0.0
    return res

def evaluate(df, score_col, label_col="label", group_col="session_id", ks=(10,), seed=0):
    rng = np.random.default_rng(seed)
    acc = {f"NDCG@{k}": [] for k in ks}
    acc["MRR@200"] = []; acc["P@1"] = []; acc["HR@10"] = []
    for _, g in df.groupby(group_col, sort=False):
        labels = g[label_col].values
        if (labels > 0).sum() == 0:
            continue
        scores = g[score_col].values.astype(float)
        tb = rng.random(len(scores))               # random tie-break
        ranked = _ranked_labels(labels, scores, tb)
        m = _metrics_from_ranked(ranked, ks)
        for key in acc: acc[key].append(m[key])
    out = {key: (float(np.mean(v)) if len(v) else 0.0) for key, v in acc.items()}
    return out, len(acc["P@1"])
