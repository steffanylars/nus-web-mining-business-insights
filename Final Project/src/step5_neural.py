"""History-conditioned neural ranking and matched history-tower ablations.

All shared parameters/preprocessing use training users only. Held-out users are
represented by their preceding history, never by an untrained user-ID embedding.
"""
import argparse
import copy
import hashlib
import json
import math
import random
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from config import (OUT, RES, SEED, HISTORY_LENGTH, EPOCHS, BATCH_SIZE, THREADS,
                    feature_columns, prepare, save_json)
from data_utils import short_mask, select_sequence
from eval_utils import evaluate

VARIANTS = ['mlp','pooled','short_only','order_only','dual','history_cf']
NAMES = {'mlp':'MLP(no towers)','pooled':'Pooled history','short_only':'Short only',
         'order_only':'Order only','dual':'DHAR(ours)','history_cf':'History-conditioned CF'}
BRAND_BUCKETS = 20000

def device():
    return torch.device('cuda' if torch.cuda.is_available() else
                        'mps' if torch.backends.mps.is_available() else 'cpu')

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def brand_bucket(value):
    if value is None or pd.isna(value) or str(value).strip() in {'','-1'}:
        return 0
    digest=hashlib.blake2b(str(value).encode(),digest_size=8).digest()
    return int.from_bytes(digest,'little')%(BRAND_BUCKETS-1)+1

def fit_preprocessing(feat,hist,meta,length):
    cols=feature_columns(feat); train=feat.split.eq('train')
    train_ids=set(feat.loc[train,'session_id'])
    train_hist=hist[hist.session_id.isin(train_ids)]
    train_wids=set(feat.loc[train,'cand_wid'])|set(train_hist.wid)
    tm=meta[meta.wid.isin(train_wids)]
    categories=sorted(c for c in tm.cate_id_3.dropna().unique() if str(c).strip())
    mu=feat.loc[train,cols].mean()
    sd=feat.loc[train,cols].std().fillna(1).replace(0,1)
    return dict(schema_version=2,feature_columns=cols,mean=mu.tolist(),std=sd.tolist(),
        category_ids={str(c):i+1 for i,c in enumerate(categories)},
        item_ids={w:i+1 for i,w in enumerate(sorted(train_wids))},
        history_length=length,cf_history_length=2*length,brand_buckets=BRAND_BUCKETS,
        brand_hash='blake2b UTF-8, 8 bytes little-endian; modulo 19999 + 1',
        sequence_policy='evenly spaced positions over each complete chronological horizon',
        short_policy='CLICK <=7d or CART <=90d',order_policy='all preceding ORD events')

def encode_frames(feat,hist,meta,prep):
    frame=feat.copy()
    cols=prep['feature_columns']
    frame[cols]=((frame[cols]-np.array(prep['mean']))/np.array(prep['std'])).astype('float32')
    if not np.isfinite(frame[cols].to_numpy()).all():
        raise ValueError('Nonfinite neural features')
    maps=meta.set_index('wid')
    cat=maps.cate_id_3.to_dict(); brand=maps.brand_id.to_dict()
    frame['_c3']=frame.cand_wid.map(cat).map(prep['category_ids']).fillna(0).astype(int)
    frame['_bi']=frame.cand_wid.map(brand).map(brand_bucket).astype(int)
    frame['_ii']=frame.cand_wid.map(prep['item_ids']).fillna(0).astype(int)
    h=hist.sort_values(['session_id','pos']).copy()
    h['_c3']=h.wid.map(cat).map(prep['category_ids']).fillna(0).astype(int)
    h['_ii']=h.wid.map(prep['item_ids']).fillna(0).astype(int)
    sequences={}; coverage={}
    for name,subset,col,length in [
        ('short',h[short_mask(h)],'_c3',prep['history_length']),
        ('order',h[h.type.eq('ORD')],'_c3',prep['history_length']),
        ('cf',h,'_ii',prep['cf_history_length'])]:
        seq={}; eligible=retained=unknown=0
        for sid,g in subset.groupby('session_id',sort=False):
            values=g[col].to_numpy()
            eligible+=len(values); unknown+=int((values==0).sum())
            # Unknown metadata/items are excluded before position sampling.
            selected=select_sequence(values[values>0],length)
            seq[sid]=selected; retained+=len(selected)
        sequences[name]=seq
        coverage[name]={'eligible_events':eligible,'unknown_events':unknown,'selected_events':retained,
                        'users_with_known_sequence':sum(bool(len(v)) for v in seq.values())}
    return frame,sequences,coverage

def pack(frame,sequences,prep,split,width=200,seed=SEED):
    cols=prep['feature_columns']; groups=frame[frame.split.eq(split)].groupby('session_id',sort=True)
    n=len(groups); nf=len(cols); L=prep['history_length']; H=prep['cf_history_length']
    arrays=dict(num=np.zeros((n,width,nf),dtype=np.float32),
        c3=np.zeros((n,width),dtype=np.int64),bi=np.zeros((n,width),dtype=np.int64),
        ii=np.zeros((n,width),dtype=np.int64),lab=np.zeros((n,width),dtype=np.int64),
        mask=np.zeros((n,width),dtype=bool),ridx=np.full((n,width),-1,dtype=np.int64),
        short=np.zeros((n,L),dtype=np.int64),order=np.zeros((n,L),dtype=np.int64),
        cf=np.zeros((n,H),dtype=np.int64))
    for row,(sid,g) in enumerate(groups):
        # Canonical order prevents source positions influencing sampling/roundoff.
        g=g.sort_values('cand_wid')
        if len(g)>width:
            if split!='train':
                raise ValueError('Evaluation must use every supplied candidate')
            rng=np.random.default_rng(np.random.SeedSequence([seed,int(sid)]))
            pos=np.flatnonzero(g.label.to_numpy()>0); neg=np.flatnonzero(g.label.to_numpy()==0)
            if len(pos)>width:
                raise ValueError('Training width is smaller than number of positives')
            selected=np.r_[pos,rng.choice(neg,width-len(pos),replace=False)]
            g=g.iloc[selected].sort_values('cand_wid')
        m=len(g)
        arrays['num'][row,:m]=g[cols].to_numpy(dtype=np.float32)
        for key,col in [('c3','_c3'),('bi','_bi'),('ii','_ii'),('lab','label')]:
            arrays[key][row,:m]=g[col].to_numpy()
        arrays['mask'][row,:m]=True; arrays['ridx'][row,:m]=g.index.to_numpy()
        for key in ['short','order','cf']:
            v=sequences[key].get(sid,[])
            arrays[key][row,:len(v)]=v
    return {k:torch.from_numpy(v) for k,v in arrays.items()}

class DHAR(nn.Module):
    def __init__(self,nc3,nb,nf,variant='dual',d=32,**unused):
        super().__init__(); self.variant=variant
        self.ec3=nn.Embedding(nc3,d,padding_idx=0); self.eb=nn.Embedding(nb,d,padding_idx=0)
        self.ps=nn.Linear(d,d); self.pl=nn.Linear(d,d)
        self.head=nn.Sequential(nn.Linear(4*d+nf,128),nn.ReLU(),nn.Dropout(.2),
            nn.Linear(128,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,1))
    def attend(self,q,seq,projection):
        emb=self.ec3(seq); mask=seq>0
        score=torch.einsum('bsd,bld->bsl',projection(q),emb)/math.sqrt(emb.size(-1))
        score=score.masked_fill(~mask[:,None,:],-1e9)
        weight=torch.softmax(score,dim=-1)*mask[:,None,:]
        weight=weight/weight.sum(-1,keepdim=True).clamp_min(1e-9)
        return torch.einsum('bsl,bld->bsd',weight,emb)
    def forward(self,num,c3,bi,ii,short,order,cf):
        qc=self.ec3(c3); qb=self.eb(bi); sc=torch.zeros_like(qc); lc=torch.zeros_like(qc)
        if self.variant in {'dual','short_only'}:
            sc=self.attend(qc,short,self.ps)
        if self.variant in {'dual','order_only'}:
            lc=self.attend(qc,order,self.pl)
        if self.variant=='pooled':
            both=torch.cat([short,order],dim=1); mask=both>0
            pool=(self.ec3(both)*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
            sc=pool[:,None,:].expand_as(qc); lc=sc
        return self.head(torch.cat([qc,qb,sc,lc,num],dim=-1)).squeeze(-1)

class HistoryCF(nn.Module):
    """GMF+MLP with a mean history encoder instead of per-session embeddings."""
    def __init__(self,ni,d=32,**unused):
        super().__init__()
        self.item=nn.Embedding(ni,d,padding_idx=0)
        self.mlp=nn.Sequential(nn.Linear(2*d,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        self.out=nn.Linear(d+32,1)
    def forward(self,num,c3,bi,ii,short,order,cf):
        mask=cf>0
        user=(self.item(cf)*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        candidate=self.item(ii); user=user[:,None,:].expand_as(candidate)
        return self.out(torch.cat([user*candidate,self.mlp(torch.cat([user,candidate],-1))],-1)).squeeze(-1)

def listwise(scores,lab,mask):
    gain=torch.tensor([0.,1.,3.,7.],device=scores.device)[lab]*mask
    target=gain/gain.sum(1,keepdim=True).clamp_min(1e-9)
    loss=-(target*F.log_softmax(scores.masked_fill(~mask,-1e9),dim=1)).sum(1)
    positive=gain.sum(1)>0
    return loss[positive].mean() if positive.any() else scores.sum()*0

def forward_batch(model,pk,indices,dev):
    return model(*[pk[k][indices].to(dev) for k in ['num','c3','bi','ii','short','order','cf']])

def predict_pack(model,pk,n_rows,dev):
    model.eval(); scores=np.zeros(n_rows,dtype=np.float32)
    with torch.no_grad():
        for a in range(0,len(pk['num']),BATCH_SIZE):
            sl=slice(a,a+BATCH_SIZE)
            value=forward_batch(model,pk,sl,dev).cpu().numpy()
            ids=pk['ridx'][sl].numpy(); mask=pk['mask'][sl].numpy()
            scores[ids[mask]]=value[mask]
    return scores

def train_one(variant,prep,frame,packs,seed,dev):
    seed_everything(seed)
    spec=dict(nc3=len(prep['category_ids'])+1,nb=prep['brand_buckets'],
              nf=len(prep['feature_columns']),ni=len(prep['item_ids'])+1,variant=variant)
    model=(HistoryCF(**spec) if variant=='history_cf' else DHAR(**spec)).to(dev)
    optimizer=torch.optim.Adam(model.parameters(),lr=.002,weight_decay=1e-6)
    best=-1.; state=None; bad=0; log=[]; start=time.time()
    val=frame[frame.split.eq('val')].copy()
    tr=packs['train']
    for epoch in range(1,EPOCHS+1):
        model.train(); perm=np.random.permutation(len(tr['num'])); total=0.; batches=0
        for a in range(0,len(perm),BATCH_SIZE):
            ix=perm[a:a+BATCH_SIZE]
            score=forward_batch(model,tr,ix,dev)
            loss=listwise(score,tr['lab'][ix].to(dev),tr['mask'][ix].to(dev))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total+=float(loss.detach()); batches+=1
        scores=predict_pack(model,packs['val'],len(frame),dev)
        val['score']=scores[val.index.to_numpy()]
        metrics,n=evaluate(val,'score'); ndcg=metrics['NDCG@10']
        log.append(dict(epoch=epoch,loss=total/max(batches,1),val_ndcg10=ndcg))
        print(f'{variant} seed={seed} epoch={epoch} val={ndcg:.5f} elapsed={time.time()-start:.0f}s',flush=True)
        if ndcg>best:
            best=ndcg; state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()}); bad=0
        else:
            bad+=1
            if bad>=3: break
    model.load_state_dict(state)
    name=f'{variant}_seed{seed}'
    torch.save({'state_dict':state,'spec':spec,'schema_version':2},OUT/f'{name}.pt')
    save_json(RES/f'training_{name}.json',dict(seed=seed,variant=variant,epochs=log,
        best_validation_ndcg10=best,training_slate_width=packs['train']['mask'].shape[1],
        evaluation_slate_width=200,batch_size=BATCH_SIZE,device=str(dev)))
    return model

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variants',nargs='+',choices=VARIANTS,default=VARIANTS)
    parser.add_argument('--seeds',nargs='+',type=int,default=[SEED])
    args=parser.parse_args()
    prepare(); torch.set_num_threads(THREADS); dev=device()
    print('device',dev,flush=True)
    feat=pd.read_parquet(OUT/'features.parquet')
    hist=pd.read_parquet(OUT/'history.parquet')
    meta=pd.read_parquet(OUT/'meta_subset.parquet')
    prep=fit_preprocessing(feat,hist,meta,HISTORY_LENGTH)
    save_json(OUT/'neural_preprocessing.json',prep)
    frame,sequences,coverage=encode_frames(feat,hist,meta,prep)
    save_json(RES/'horizon_coverage.json',coverage)
    # Same fixed negative sample and complete eval slates for every architecture.
    packs={s:pack(frame,sequences,prep,s,64 if s=='train' else 200)
           for s in ['train','val','test']}
    results={}; predictions=feat.loc[feat.split.eq('test'),['session_id','cand_wid','label']].copy()
    for seed in args.seeds:
        for variant in args.variants:
            model=train_one(variant,prep,frame,packs,seed,dev)
            key=NAMES[variant] if len(args.seeds)==1 else f'{NAMES[variant]} seed{seed}'
            test_scores=predict_pack(model,packs['test'],len(frame),dev)
            col=f'{variant}_score' if len(args.seeds)==1 else f'{variant}_seed{seed}_score'
            predictions[col]=test_scores[predictions.index.to_numpy()]
            m,n=evaluate(predictions,col); results[key]=m
            print('TEST',key,m,flush=True)
            val=frame[frame.split.eq('val')].copy()
            score=predict_pack(model,packs['val'],len(frame),dev)
            val['score']=score[val.index.to_numpy()]
            results[key+'_val']=evaluate(val,'score')[0]
            # Incremental saves allow a long CPU run to be resumed/reviewed.
            predictions.to_parquet(OUT/'test_pred_neural.parquet',index=False)
            save_json(RES/'results_neural.json',results)
            del model
    print('Neural variants complete',flush=True)

if __name__=='__main__':
    main()
