"""Macro query metrics with order-invariant deterministic tie breaking."""
import numpy as np
import pandas as pd

def tie_keys(df, seed=0, group_col='session_id'):
    if 'cand_wid' not in df:
        raise ValueError('Candidate IDs are required for order-independent tie breaking')
    keys = df[[group_col,'cand_wid']].astype(str).assign(_seed=str(seed))
    return pd.util.hash_pandas_object(keys, index=False).to_numpy(dtype=np.uint64)

def ranked_order(group, score_col, seed=0):
    scores = group[score_col].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError('Ranking scores must be finite')
    return np.lexsort((tie_keys(group,seed), -scores))

def _ranked_labels(labels, scores, tiebreak):
    return np.asarray(labels)[np.lexsort((tiebreak, -np.asarray(scores)))]

def _dcg(rels, k):
    rels = np.asarray(rels, dtype=float)[:k]
    return float(((2**rels-1) / np.log2(np.arange(2,len(rels)+2))).sum())

def _metrics_from_ranked(ranked, ks):
    ranked = np.asarray(ranked)
    ideal = np.sort(ranked)[::-1]
    result = {}
    for k in ks:
        idcg = _dcg(ideal,k)
        result[f'NDCG@{k}'] = _dcg(ranked,k)/idcg if idcg else 0.
    nz = np.flatnonzero(ranked[:200] > 0)
    result['MRR@200'] = 1./(nz[0]+1) if nz.size else 0.
    result['P@1'] = float(ranked[0] > 0) if len(ranked) else 0.
    result['HR@10'] = float((ranked[:10] > 0).any())
    return result

def per_query(df, score_col, label_col='label', group_col='session_id', ks=(10,), seed=0):
    if df.duplicated([group_col,'cand_wid']).any():
        raise ValueError('Duplicate query/candidate keys')
    records = []
    # Compute all hashes once, not once per group.
    frame = df.copy()
    frame['_tie'] = tie_keys(frame,seed,group_col)
    if not np.isfinite(frame[score_col].to_numpy(dtype=float)).all():
        raise ValueError('Ranking scores must be finite')
    for sid,g in frame.groupby(group_col,sort=True):
        labels = g[label_col].to_numpy()
        if not (labels > 0).any():
            continue
        ranked = _ranked_labels(labels,g[score_col].to_numpy(),g['_tie'].to_numpy())
        records.append({group_col:sid, **_metrics_from_ranked(ranked,ks)})
    return pd.DataFrame(records, columns=[group_col,*[f'NDCG@{k}' for k in ks],
                                         'MRR@200','P@1','HR@10'])

def evaluate(df, score_col, label_col='label', group_col='session_id', ks=(10,), seed=0):
    rows = per_query(df,score_col,label_col,group_col,ks,seed)
    cols = [c for c in rows if c!=group_col]
    return {c:float(rows[c].mean()) if len(rows) else 0. for c in cols},len(rows)

def lightgbm_metric(frame):
    """Same positive-query policy and tie keys as final evaluation, vectorized."""
    groups = list(frame.groupby('session_id',sort=False).indices.values())
    width = max(map(len,groups))
    idx = np.zeros((len(groups),width),dtype=int)
    mask = np.zeros_like(idx,dtype=bool)
    for i,ids in enumerate(groups):
        idx[i,:len(ids)] = ids; mask[i,:len(ids)] = True
    labels = frame.label.to_numpy()[idx]
    labels = np.where(mask,labels,0)
    ties = tie_keys(frame)[idx]
    gain = 2.**labels-1
    discount = 1./np.log2(np.arange(2,min(width,10)+2))
    ideal = np.sort(gain,axis=1)[:,::-1][:,:10]
    idcg = (ideal*discount).sum(1); positive = idcg>0
    def metric(prediction, dataset):
        scores = np.where(mask,np.asarray(prediction)[idx],-np.inf)
        order = np.lexsort((ties,-scores),axis=1)[:,:10]
        ranked = np.take_along_axis(gain,order,axis=1)
        value = ((ranked*discount).sum(1)[positive]/idcg[positive]).mean() if positive.any() else 0.
        return 'positive_query_ndcg10',float(value),True
    return metric
