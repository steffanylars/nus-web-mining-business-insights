"""Build a fixed user split, complete histories and a raw-catalogue metadata subset."""
import csv
import hashlib
import sys
import time
import numpy as np
import pandas as pd
from config import OUT, RES, SRC_USER, SRC_PRODUCT, SEED, SCHEMA_VERSION, prepare, save_json
from data_utils import split_list, elapsed_days

def build_records(records, selected):
    candidates, histories, sessions = [], [], []
    selected = set(selected)
    sid = 0
    for source_row, row in enumerate(records):
        if source_row not in selected:
            continue
        cw = split_list(row['candidate_wid_list'])
        labs = [int(float(x)) for x in split_list(row['candidate_label_list'])]
        hw = split_list(row['history_wid_list'])
        ht = split_list(row['history_type_list'])
        hq = split_list(row['history_qry_list'])
        if not cw or len(cw) != len(labs) or len(set(cw)) != len(cw):
            raise ValueError(f'Invalid candidate alignment at source row {source_row}')
        if len(hw) != len(ht) or len(hw) != len(hq) or any(x not in {'CLICK','CART','ORD','FLW'} for x in ht):
            raise ValueError(f'Invalid history alignment at source row {source_row}')
        if any(x not in {0,1,2,3} for x in labs):
            raise ValueError(f'Invalid labels at source row {source_row}')
        age = elapsed_days([float(x) for x in split_list(row['history_time_list'])], len(hw))
        candidates.extend((sid,row['query'],w,lab) for w,lab in zip(cw,labs))
        histories.extend((sid,pos,w,t,q,float(d)) for pos,(w,t,q,d) in enumerate(zip(hw,ht,hq,age)))
        sessions.append((sid,source_row,row['query'],len(cw),len(hw)))
        sid += 1
    cand = pd.DataFrame(candidates, columns=['session_id','query','cand_wid','label'])
    hist = pd.DataFrame(histories, columns=['session_id','pos','wid','type','qry','elapsed_days'])
    sess = pd.DataFrame(sessions, columns=['session_id','source_row_id','query','slate_size','n_hist'])
    perm = np.random.default_rng(SEED).permutation(len(sess))
    a = int(.70*len(sess)); b = a + int(.15*len(sess))
    split = np.empty(len(sess), dtype=object)
    split[perm[:a]], split[perm[a:b]], split[perm[b:]] = 'train', 'val', 'test'
    sess['split'] = split
    return cand, hist, sess

def main():
    prepare(); start = time.time()
    n_users = int(sys.argv[1]) if len(sys.argv)>1 else 20000
    if n_users < 10:
        raise ValueError('Use at least 10 users for a three-way split')
    with SRC_USER.open('rb') as f:
        n_rows = sum(1 for _ in f)-1
    idx = np.sort(np.random.default_rng(SEED).choice(n_rows, min(n_rows,n_users), replace=False))
    with SRC_USER.open(encoding='utf-8', newline='') as f:
        cand,hist,sess = build_records(csv.DictReader(f, delimiter='\t'), idx.tolist())
    print(f'Built {len(sess):,} users; {len(hist):,} complete history events', flush=True)
    refs = set(cand.cand_wid) | set(hist.wid)
    metadata = []
    with SRC_PRODUCT.open('rb') as f:
        headers = next(f).decode('utf-8').rstrip('\r\n').split('\t')
        for line in f:
            wid = line.split(b'\t',1)[0].decode('utf-8')
            if wid in refs:
                fields = line.decode('utf-8').rstrip('\r\n').split('\t')
                if len(fields) != len(headers):
                    raise ValueError(f'Invalid metadata alignment for {wid}')
                metadata.append(fields)
    meta = pd.DataFrame(metadata, columns=headers).drop_duplicates('wid')
    if not {'cate_name_3','brand_name','name'}.issubset(meta.columns):
        raise ValueError('Raw metadata requires names as well as IDs')
    for name,frame in [('candidates',cand),('history',hist),('sessions',sess),('meta_subset',meta)]:
        frame.to_parquet(OUT/f'{name}.parquet', index=False)
    membership = sess[['session_id','source_row_id','split']].to_csv(index=False)
    save_json(RES/'dataset_manifest.json', dict(schema_version=SCHEMA_VERSION, seed=SEED,
        source_users=n_rows, sampled_users=len(sess), candidate_rows=len(cand), history_rows=len(hist),
        history_policy='all preceding events; independent horizon selection occurs in neural packing',
        split_counts=sess.groupby('split').size().to_dict(),
        split_sha256=hashlib.sha256(membership.encode()).hexdigest(),
        candidate_metadata_coverage=float(cand.cand_wid.isin(set(meta.wid)).mean()),
        raw_files={p.name: {'bytes':p.stat().st_size} for p in [SRC_USER,SRC_PRODUCT]}))
    print(f'Dataset ready in {OUT} ({time.time()-start:.0f}s)', flush=True)

if __name__ == '__main__':
    main()
