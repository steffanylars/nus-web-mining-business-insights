"""Read-only audit of pinned source, local raw data and committed predictions."""
from pathlib import Path
import ast, csv, json, sys, collections, os, subprocess
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PINNED = '3d6b62b90fbdee102331f2aaf099be0ec52cb5e8'
def original_source(name):
    return subprocess.check_output(['git','show',f'{PINNED}:Final Project/src/{name}'],
                                   cwd=ROOT).decode('utf-8-sig')
scope_eval = {}
exec(compile(original_source('eval_utils.py'), '<pinned evaluation>', 'exec'), scope_eval)
evaluate = scope_eval['evaluate']
_ranked_labels = scope_eval['_ranked_labels']
_metrics_from_ranked = scope_eval['_metrics_from_ranked']
SRC = ROOT / 'Final Project/src'
ART = ROOT / 'Final Project/artifacts'
RAW = Path(os.environ.get('JD_RAW_DIR', ROOT / 'Final Project' / 'data'))
OUT = Path(__file__).parent
out = {}

# Extract only tok(), avoiding the source module's hard-coded output path.
tree = ast.parse(original_source('step3_features.py'))
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'tok')
scope = {}
exec(compile(ast.Module(body=[fn], type_ignores=[]), '<source tok>', 'exec'), scope)
tok = scope['tok']
out['tokenizer_example'] = {'actual': sorted(tok('12\x1832\x1856')), 'expected': ['12','32','56']}
assert len(tok('12\x1832\x1856')) == 1
assert np.cumsum(np.array([0,12,4,5])[::-1])[::-1][1:].tolist() == [21,9,5]

sess = pd.read_parquet(ART/'model/sessions.parquet')
lp = pd.read_parquet(ART/'model/test_pred_lgb.parquet')
npred = pd.read_parquet(ART/'model/test_pred_neural.parquet')
out['sessions'] = sess.groupby('split').size().to_dict()
out['prediction_integrity'] = {'lgb_rows':len(lp), 'neural_rows':len(npred),
    'duplicate_keys':int(lp.duplicated(['session_id','cand_wid']).sum()),
    'test_ids_match':set(lp.session_id)==set(sess.loc[sess.split=='test','session_id'])}
joined=lp.merge(npred,on=['session_id','cand_wid'],suffixes=('_lgb','_neural'),validate='one_to_one')
assert len(joined)==len(lp)==len(npred)
assert (joined.label_lgb==joined.label_neural).all()
out['recomputed_metrics']={}
for df,col in [(lp,'lgb_score'),(npred,'dhar_score'),(npred,'ncf_score')]:
    m,n=evaluate(df,col)
    out['recomputed_metrics'][col]={'metrics':m,'n':n}

# Paired query-level uncertainty, using the repository's tie-breaking convention.
per=[]
for frame,score in [(lp,'lgb_score'),(npred,'dhar_score')]:
    rng=np.random.default_rng(0); vals={}
    for sid,g in frame.groupby('session_id',sort=False):
        labels=g.label.to_numpy()
        if not (labels>0).any(): continue
        ranked=_ranked_labels(labels,g[score].to_numpy(),rng.random(len(g)))
        vals[sid]=_metrics_from_ranked(ranked,(10,))['NDCG@10']
    per.append(np.array([vals[sid] for sid in sorted(vals)]))
delta=per[1]-per[0]; rng=np.random.default_rng(42)
boot=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(2000)])
out['paired_bootstrap_saved_predictions']={'delta':float(delta.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),
    'caveat':'Conditional on these saved models and this sample; does not validate features or isolate architecture.'}
print('Prediction checks complete',flush=True)

N=173831
idx=np.random.default_rng(42).choice(N,size=20000,replace=False); idx.sort()
selected=set(idx.tolist()); content_rows=set(idx[:200].tolist())
edges=np.array([1/1440,1/24,1,7,30,90,180,365,np.inf])
types=['CLICK','CART','ORD','FLW']; counts=np.zeros((9,4),dtype=np.int64)
all_orders=kept_orders=old_orders=kept_old_orders=0
users_orders=users_kept_orders=users_lost_all_orders=0
hist_rows=0; sample_cands=0; content=[]; refs=set(); rows=0; alignment=0
with (RAW/'user_behavior_data.txt').open(encoding='utf-8',newline='') as f:
    for i,r in enumerate(csv.DictReader(f,delimiter='\t')):
        acts=np.array(r['history_type_list'].split('_'))
        gaps=np.fromstring(r['history_time_list'],sep='_')
        rows+=1
        if len(gaps)!=len(acts)+1: alignment+=1; continue
        age=np.cumsum(gaps[::-1])[::-1][1:]/86400
        bins=np.digitize(age,edges,right=False)
        for t,typ in enumerate(types): counts[:,t]+=np.bincount(bins[acts==typ],minlength=9)
        if i in selected:
            keep=np.sort(np.argsort(age)[:60]); isord=acts=='ORD'; isold=isord&(age>90)
            all_orders+=int(isord.sum()); kept_orders+=int(isord[keep].sum())
            old_orders+=int(isold.sum()); kept_old_orders+=int(isold[keep].sum())
            users_orders+=int(isord.any()); users_kept_orders+=int(isord[keep].any())
            users_lost_all_orders+=int(isord.any() and not isord[keep].any())
            hist_rows+=len(keep); sample_cands+=len(r['candidate_wid_list'].split('_'))
        if i in content_rows:
            for w in r['candidate_wid_list'].split('_'):
                content.append((w,r['query'])); refs.add(w)
out['raw_temporal']={'users':rows,'alignment_failures':alignment,'types':types,'counts_by_bin':counts.tolist()}
out['sample_history_cap']={'users':20000,'kept_history_rows':hist_rows,'candidate_rows':sample_cands,
    'orders_before':all_orders,'orders_after':kept_orders,'orders_older90_before':old_orders,
    'orders_older90_after':kept_old_orders,'users_with_orders_before':users_orders,
    'users_with_orders_after':users_kept_orders,'users_losing_all_orders':users_lost_all_orders}
print('Full history scan complete',out['sample_history_cap'],flush=True)

# Scan raw catalogue once, retain metadata for a small deterministic query sample.
meta={}
with (RAW/'product_meta_data.txt').open('rb') as f:
    header=next(f).decode('utf-8').rstrip('\r\n').split('\t')
    for line in f:
        wid=line.split(b'\t',1)[0].decode('utf-8')
        if wid in refs:
            meta[wid]=dict(zip(header,line.decode('utf-8').rstrip('\r\n').split('\t')))
def correct(s): return set(s.split('\x18')) if s else set()
old=[]; fixed=[]; qlens=[]
for w,q in content:
    if w not in meta: continue
    name=meta[w]['name']
    old.append(len(tok(q)&tok(name))); fixed.append(len(correct(q)&correct(name)))
    qlens.append((len(tok(q)),len(correct(q))))
out['raw_content_sample']={'query_records':200,'pairs':len(content),'metadata_matched_pairs':len(old),
    'source_mean_overlap':float(np.mean(old)),'correct_mean_overlap':float(np.mean(fixed)),
    'source_nonzero_overlap_pairs':int(np.count_nonzero(old)),
    'correct_nonzero_overlap_pairs':int(np.count_nonzero(fixed)),
    'source_mean_query_unique_tokens':float(np.mean([q[0] for q in qlens])),
    'correct_mean_query_unique_tokens':float(np.mean([q[1] for q in qlens]))}
(OUT/'verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2),flush=True)
