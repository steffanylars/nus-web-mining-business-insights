"""Verify corrected saved artifacts, fixed split, metrics and standalone model reload."""
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Final Project/src'))
from config import BASE,OUT,RES,save_json
from eval_utils import evaluate
from predict_neural import predict

def main():
    old=pd.read_parquet(BASE/'artifacts/model/sessions.parquet')
    new=pd.read_parquet(OUT/'sessions.parquet')
    assert old[['session_id','query','slate_size','n_hist','split']].equals(
        new[['session_id','query','slate_size','n_hist','split']])
    hist=pd.read_parquet(OUT/'history.parquet')
    assert int(new.n_hist.sum())==len(hist)==3091153
    feat=pd.read_parquet(OUT/'features.parquet')
    assert (feat.query_len>1).any()
    assert (feat.ov_cat>0).any()
    assert (feat.h_len.max()>60)
    lp=pd.read_parquet(OUT/'test_pred_lgb.parquet')
    npred=pd.read_parquet(OUT/'test_pred_neural.parquet')
    expected=json.loads((RES/'results_neural.json').read_text())
    names={'mlp':'MLP(no towers)','pooled':'Pooled history','short_only':'Short only',
           'order_only':'Order only','dual':'DHAR(ours)','history_cf':'History-conditioned CF'}
    recomputed={}
    for prefix,name in names.items():
        score=prefix+'_score'
        assert score in npred,score
        m,n=evaluate(npred,score)
        assert n==2965
        for k,v in m.items():
            assert abs(v-expected[name][k])<1e-12,(name,k)
        recomputed[name]=m
    recorded=json.loads((RES/'results_baselines_lgb.json').read_text())
    m,n=evaluate(lp,'lgb_score')
    for k,v in m.items():
        assert abs(v-recorded['LambdaMART(features)'][k])<1e-12
    expected_rows=feat[feat.split.eq('test')][['session_id','cand_wid','label']]
    for pred in [lp,npred]:
        joined=pred.merge(expected_rows,on=['session_id','cand_wid'],validate='one_to_one',suffixes=('_a','_b'))
        assert len(joined)==len(expected_rows)==len(pred)
        assert (joined.label_a==joined.label_b).all()
    ids=sorted(npred.session_id.unique())[:8]
    sample=feat[feat.session_id.isin(ids)].drop(columns=['label'])
    h=hist[hist.session_id.isin(ids)]
    meta=pd.read_parquet(OUT/'meta_subset.parquet')
    refs=set(sample.cand_wid)|set(h.wid)
    meta=meta[meta.wid.isin(refs)]
    scored=predict(OUT/'dual_seed42.pt',OUT/'neural_preprocessing.json',sample,h,meta)
    joined=scored.merge(npred[['session_id','cand_wid','dual_score']],on=['session_id','cand_wid'],validate='one_to_one')
    max_error=float(np.abs(joined.score-joined.dual_score).max())
    assert max_error<2e-5,max_error
    versions={n:importlib.metadata.version(n) for n in
        ['numpy','pandas','pyarrow','scipy','scikit-learn','lightgbm','torch','matplotlib','joblib']}
    output=dict(fixed_original_split=True,complete_history_events=len(hist),test_candidates=len(lp),
        positive_test_queries=n,neural_metrics_match_saved_json=True,standalone_reload_candidates=len(scored),
        standalone_reload_max_absolute_error=max_error,recomputed_neural_metrics=recomputed,
        versions=versions,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted((BASE/'src').glob('*.py'))})
    save_json(RES/'verified_run.json',output)
    print(json.dumps(output,indent=2))

if __name__=='__main__':
    main()
