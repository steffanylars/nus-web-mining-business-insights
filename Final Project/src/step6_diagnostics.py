"""Full-slate subgroup ranks, paired uncertainty and matched feature ablation."""
import argparse
import numpy as np
import pandas as pd
from config import OUT, RES, SEED, feature_columns, prepare, save_json
from eval_utils import evaluate, ranked_order, per_query
from step4_baselines_lgbm import fit_ranker, LGB_PARAMS

TEMPORAL_FEATURES=['aff_cat_decay','aff_brand_decay','short_cat_hit','short_brand_hit',
    'long_cat_hit','long_brand_hit','h_min_elapsed','h_n_recent7','h_n_ord','h_n_cart','h_n_click']

def positive_rank_records(pred,score,history_seen,training_candidates):
    records=[]
    for sid,g in pred.groupby('session_id',sort=True):
        order=ranked_order(g,score); ranks=np.empty(len(g),dtype=int)
        ranks[order]=np.arange(1,len(g)+1)
        for w,lab,rk in zip(g.cand_wid,g.label,ranks):
            if lab>0:
                records.append(dict(session_id=int(sid),cand_wid=w,label=int(lab),rank=int(rk),
                    training_history_seen=w in history_seen,
                    training_candidate_seen=w in training_candidates))
    return pd.DataFrame(records)

def bootstrap_difference(first,second,score_first,score_second,repetitions=2000):
    a=per_query(first,score_first).set_index('session_id')['NDCG@10']
    b=per_query(second,score_second).set_index('session_id')['NDCG@10']
    if set(a.index)!=set(b.index):
        raise ValueError('Paired comparison requires identical evaluable queries')
    delta=b.sort_index().to_numpy()-a.sort_index().to_numpy()
    rng=np.random.default_rng(SEED)
    boot=[rng.choice(delta,len(delta),replace=True).mean() for _ in range(repetitions)]
    return dict(delta=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),
        positive_queries=len(delta),resamples=repetitions,seed=SEED,
        interpretation='conditional on fixed saved models; not training-seed uncertainty')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--skip-feature-ablation',action='store_true')
    args=parser.parse_args()
    prepare()
    feat=pd.read_parquet(OUT/'features.parquet')
    sess=pd.read_parquet(OUT/'sessions.parquet')
    hist=pd.read_parquet(OUT/'history.parquet')
    lp=pd.read_parquet(OUT/'test_pred_lgb.parquet')
    npred=pd.read_parquet(OUT/'test_pred_neural.parquet')
    keys=['session_id','cand_wid']
    merged=lp[keys+['label']].merge(npred[keys+['label']],on=keys,validate='one_to_one',suffixes=('_a','_b'))
    if len(merged)!=len(lp) or len(merged)!=len(npred) or not (merged.label_a==merged.label_b).all():
        raise ValueError('Prediction populations differ')
    train_ids=set(sess.loc[sess.split.eq('train'),'session_id'])
    history_seen=set(hist.loc[hist.session_id.isin(train_ids),'wid'])
    candidate_seen=set(feat.loc[feat.split.eq('train'),'cand_wid'])
    models=[('LambdaMART',lp,'lgb_score')]
    models.extend((c.removesuffix('_score'),npred,c) for c in npred if c.endswith('_score'))
    segments=[]; rank_tables=[]; recalls=[]
    history_bins=pd.cut(sess.n_hist,[-1,2,20,100,np.inf],
        labels=['limited(0-2)','short(3-20)','medium(21-100)','long(>100)'])
    history_map=dict(zip(sess.session_id,history_bins.astype(str)))
    for name,pred,col in models:
        frame=pred.copy(); frame['history_group']=frame.session_id.map(history_map)
        for group,g in frame.groupby('history_group',sort=True):
            m,n=evaluate(g,col)
            segments.append(dict(model=name,history_group=group,positive_queries=n,
                all_queries=int(g.session_id.nunique()),**m))
        rr=positive_rank_records(pred,col,history_seen,candidate_seen)
        rr['model']=name; rank_tables.append(rr)
        for dimension in ['training_history_seen','training_candidate_seen']:
            for seen,g in rr.groupby(dimension):
                by_query=g.groupby('session_id')['rank'].apply(lambda x:float((x<=10).mean()))
                recalls.append(dict(model=name,dimension=dimension,seen=bool(seen),
                    positive_items=len(g),queries_with_subgroup_positive=len(by_query),
                    macro_recall10=float(by_query.mean()),micro_recall10=float(g['rank'].le(10).mean()),
                    mean_full_slate_rank=float(g['rank'].mean()),median_full_slate_rank=float(g['rank'].median())))
    pd.DataFrame(segments).to_csv(RES/'segment_metrics.csv',index=False)
    pd.DataFrame(recalls).to_csv(RES/'positive_subgroup_full_slate.csv',index=False)
    allr=pd.concat(rank_tables,ignore_index=True)
    allr.groupby(['model','label'])['rank'].agg(['mean','median','count']).to_csv(RES/'rank_by_type.csv')
    if 'dual_score' in npred:
        save_json(RES/'paired_bootstrap.json',dict(
            dual_minus_lgb=bootstrap_difference(lp,npred,'lgb_score','dual_score'),
            dual_minus_mlp=bootstrap_difference(npred,npred,'mlp_score','dual_score'),
            dual_minus_pooled=bootstrap_difference(npred,npred,'pooled_score','dual_score')))
    if not args.skip_feature_ablation:
        # Full arm is the headline model, not an unrelated retraining configuration.
        cols=feature_columns(feat)
        booster=fit_ranker(feat,[c for c in cols if c not in TEMPORAL_FEATURES])
        test=feat[feat.split.eq('test')].copy()
        test['score']=booster.predict(test[[c for c in cols if c not in TEMPORAL_FEATURES]])
        full=evaluate(lp,'lgb_score')[0]; without=evaluate(test,'score')[0]
        save_json(RES/'ablation.json',dict(full=full,no_temporal=without,
            delta_NDCG10=full['NDCG@10']-without['NDCG@10'],removed_features=TEMPORAL_FEATURES,
            params=LGB_PARAMS,interpretation='temporal/action feature bundle, not a DHAR tower ablation'))
    save_json(RES/'diagnostic_protocol.json',dict(
        subgroup_ranks='rank every candidate first; measure positive subgroup ranks/recall in original full slate',
        history_seen='present in complete training histories',
        candidate_seen='present in training candidate rows, regardless of label',
        low_history='limited history, not proof of a new user',
        no_zero_history_claim='no natural zero-history benchmark is claimed'))
    print('Full-slate diagnostics and feature ablation complete',flush=True)

if __name__=='__main__':
    main()
