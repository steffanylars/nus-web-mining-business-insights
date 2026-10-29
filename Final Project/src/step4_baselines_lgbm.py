"""Training-only SVD with held-out fold-in, baselines and LambdaMART."""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import joblib
from config import OUT, RES, SEED, THREADS, feature_columns, prepare, save_json
from eval_utils import evaluate, tie_keys, lightgbm_metric

LGB_PARAMS = dict(objective='lambdarank', metric='None', learning_rate=.05,
    num_leaves=63, min_data_in_leaf=200, feature_fraction=.9,
    bagging_fraction=.9, bagging_freq=1, label_gain=[0,1,3,7],
    verbosity=-1, seed=SEED, num_threads=THREADS, deterministic=True, force_col_wise=True)

def fit_ranker(feat, columns):
    tr = feat[feat.split=='train'].sort_values(['session_id','cand_wid']).reset_index(drop=True)
    va = feat[feat.split=='val'].sort_values(['session_id','cand_wid']).reset_index(drop=True)
    groups = lambda d:d.groupby('session_id',sort=False).size().to_numpy()
    train = lgb.Dataset(tr[columns], label=tr.label, group=groups(tr))
    valid = lgb.Dataset(va[columns], label=va.label, group=groups(va), reference=train)
    return lgb.train(LGB_PARAMS,train,num_boost_round=600,valid_sets=[valid],
        feval=lightgbm_metric(va),
        callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(100)])

def fold_in_svd(hist, train_ids, n_sessions, dimensions=64):
    """Fit shared item factors on training histories, transform other histories only."""
    train_hist = hist[hist.session_id.isin(train_ids)]
    items = {w:i for i,w in enumerate(sorted(train_hist.wid.unique()))}
    mapped = hist.wid.map(items); keep = mapped.notna()
    matrix = csr_matrix((np.ones(int(keep.sum()),dtype=np.float32),
        (hist.loc[keep,'session_id'].to_numpy(),mapped[keep].astype(int).to_numpy())),
        shape=(n_sessions,len(items)),dtype=np.float32)
    k = min(dimensions,len(train_ids)-1,len(items)-1)
    if k < 1:
        raise ValueError('Insufficient training users/items for SVD')
    svd = TruncatedSVD(n_components=k,random_state=SEED)
    svd.fit(matrix[sorted(train_ids)])
    return items,svd,svd.transform(matrix)

def main():
    prepare(); start=time.time()
    feat=pd.read_parquet(OUT/'features.parquet')
    hist=pd.read_parquet(OUT/'history.parquet')
    sess=pd.read_parquet(OUT/'sessions.parquet')
    test=feat[feat.split=='test'].copy()
    results={}; detail={}
    def report(name,df,col):
        m,n=evaluate(df,col); results[name]=m
        detail[name]={'queries_with_positive':n,'all_queries':int(df.session_id.nunique())}
        print(name,m,flush=True)
    test['random_score']=tie_keys(test,SEED+1).astype(float)/np.iinfo(np.uint64).max
    test['content_score']=test.ov_name+test.ov_cat+.5*test.ov_brand+test.jacc_name
    report('Random',test,'random_score')
    report('Popularity',test,'pop_wid')
    report('Content/Relevance-only',test,'content_score')
    train_ids=set(sess.loc[sess.split=='train','session_id'])
    items,svd,user_factors=fold_in_svd(hist,train_ids,int(sess.session_id.max())+1)
    factors=svd.components_.T
    mapped=test.cand_wid.map(items)
    scores=np.zeros(len(test),dtype=np.float32); warm=mapped.notna().to_numpy()
    ids=mapped.fillna(0).astype(int).to_numpy()
    # Bounded temporary arrays avoid duplicating a whole candidate x factor matrix.
    sids=test.session_id.to_numpy()
    for a in range(0,len(test),20000):
        b=min(a+20000,len(test)); ok=warm[a:b]
        block=np.zeros(b-a,dtype=np.float32)
        block[ok]=np.einsum('ij,ij->i',user_factors[sids[a:b][ok]],factors[ids[a:b][ok]])
        scores[a:b]=block
    test['svd_score']=scores
    report('SVD(train-only fold-in)',test,'svd_score')
    joblib.dump({'svd':svd,'item_ids':items},OUT/'svd.joblib',compress=3)
    columns=feature_columns(feat)
    booster=fit_ranker(feat,columns)
    test['lgb_score']=booster.predict(test[columns])
    report('LambdaMART(features)',test,'lgb_score')
    for split in ['train','val']:
        frame=feat[feat.split==split].copy()
        frame['lgb_score']=booster.predict(frame[columns])
        report(f'LambdaMART(features)_{split}',frame,'lgb_score')
        report(f'Popularity_{split}',frame,'pop_wid')
    pd.DataFrame({'feature':columns,'gain':booster.feature_importance('gain')}).sort_values(
        'gain',ascending=False).to_csv(RES/'lgb_feature_importance.csv',index=False)
    booster.save_model(str(OUT/'lgb_model.txt'))
    test[['session_id','cand_wid','label','lgb_score','svd_score','random_score',
          'content_score','pop_wid']].to_parquet(OUT/'test_pred_lgb.parquet',index=False)
    save_json(OUT/'lgb_preprocessing.json',{'feature_columns':columns,'schema_version':2})
    save_json(RES/'results_baselines_lgb.json',results)
    save_json(RES/'baseline_protocol.json',dict(query_counts=detail,seed=SEED,
        svd_train_history_item_coverage=float(warm.mean()),lgb_parameters=LGB_PARAMS,
        best_iteration=booster.best_iteration,
        evaluation='macro full-slate metrics, positive queries only, deterministic ID tie keys',
        svd='train-history item factors; held-out preceding history folded into fixed factors'))
    print(f'Step 4 complete ({time.time()-start:.0f}s)',flush=True)

if __name__=='__main__':
    main()
