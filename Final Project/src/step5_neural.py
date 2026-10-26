"""
Step 5: Neural models (PyTorch, Apple MPS) -- VECTORISED padded-slate training.

(C) Dual-Horizon Attention Ranker (DHAR) -- our architecture contribution.
    Motivated by the Step-1 temporal finding (clicks 7d / carts 90d / orders 1yr
    retention => interaction TYPE is confounded with RECENCY; fields play DISTINCT roles):
      * SHORT-TERM INTENT tower : candidate-category attention over RECENT (<=7d) history categories
      * LONG-TERM PREF  tower   : candidate-category attention over ORDER-history categories
      * engineered content/popularity features fused in an MLP head
    Trained listwise (softmax cross-entropy with ordinal label gains) over each query slate.
(B) NCF baseline (GMF+MLP) on (session, item) ids -> exposes the CF cold ceiling.

Batching: each session's slate is padded/sub-sampled to a fixed width so many slates
are processed as one [B, S, ...] tensor (fast on MPS). Positives are always kept.
"""
import time, json, math
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from eval_utils import evaluate

OUT = Path("/Users/steff/Documents/GitHub/bt4212_finalProj/Final Project/artifacts/model")
RES = Path("/Users/steff/Documents/GitHub/bt4212_finalProj/Final Project/artifacts")
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(42); np.random.seed(42)
LG = np.array([0.,1.,3.,7.], dtype=np.float32)   # ordinal label gains
L = 20          # history seq length per horizon
S_TRAIN = 64    # slate width for training (keep positives + sampled negatives)
S_EVAL = 200    # slate width for eval (full)

def main():
    t0=time.time(); print("device:", DEV, flush=True)
    feat = pd.read_parquet(OUT/"features.parquet")
    hist = pd.read_parquet(OUT/"history.parquet")
    meta = pd.read_parquet(OUT/"meta_subset.parquet").set_index("wid")
    cate3 = meta["cate_id_3"].to_dict(); brand = meta["brand_id"].to_dict()
    c3map = {c:i+1 for i,c in enumerate(pd.Series(list(cate3.values())).dropna().unique())}
    NC3=len(c3map)+1; NB=20000
    def bh(b): return 0 if (b is None or str(b)=="nan") else (hash(b)%(NB-1))+1
    print(f"vocab c3={NC3} ({time.time()-t0:.0f}s)", flush=True)

    hist=hist.copy()
    hist["c3i"]=hist["wid"].map(cate3).map(lambda c:c3map.get(c,0)).fillna(0).astype(int)
    rec=hist[hist.elapsed_days<=7]; ordh=hist[hist.type=="ORD"]
    def seqmap(df):
        d={}
        for sid,g in df.groupby("session_id"): d[sid]=g["c3i"].values[-L:]
        return d
    short_seq=seqmap(rec); long_seq=seqmap(ordh)

    num_cols=[c for c in feat.columns if c not in {"session_id","cand_wid","label","split"}]
    # standardise using TRAIN statistics only (no transductive leakage into val/test)
    tr_mask=feat["split"]=="train"
    mu=feat.loc[tr_mask,num_cols].mean(); sd=feat.loc[tr_mask,num_cols].std().replace(0,1)
    feat[num_cols]=(feat[num_cols]-mu)/sd
    feat["cand_c3i"]=feat["cand_wid"].map(cate3).map(lambda c:c3map.get(c,0)).fillna(0).astype(int)
    feat["cand_bi"]=feat["cand_wid"].map(brand).map(bh).astype(int)
    item_ids={w:i+1 for i,w in enumerate(feat["cand_wid"].unique())}
    feat["ii"]=feat["cand_wid"].map(item_ids).astype(int)
    NI=len(item_ids)+1; NU=int(feat.session_id.max())+1
    Fn=len(num_cols)

    # ---- pack each session into fixed-width padded arrays ----
    def pack(split, S, keep_pos=True):
        d=feat[feat.split==split]
        sids=[]; NUM=[]; C3=[]; BI=[]; II=[]; LAB=[]; MASK=[]; SS=[]; LS=[]; RIDX=[]
        rng=np.random.default_rng(42)
        for sid,g in d.groupby("session_id",sort=False):
            idx=g.index.values; lab=g["label"].values
            if len(idx)>S:
                pos=np.where(lab>0)[0]; neg=np.where(lab==0)[0]
                nneg=max(S-len(pos),0)
                sel=np.concatenate([pos, rng.choice(neg,min(nneg,len(neg)),replace=False)]) if keep_pos \
                    else rng.choice(len(idx),S,replace=False)
                sel=sel[:S]
            else:
                sel=np.arange(len(idx))
            m=len(sel)
            pad=S-m
            def padcol(a,fill=0): return np.concatenate([a, np.full(pad,fill,dtype=a.dtype)]) if pad>0 else a
            sids.append(sid)
            NUM.append(np.vstack([g[num_cols].values[sel], np.zeros((pad,Fn),dtype=np.float32)]) if pad>0 else g[num_cols].values[sel])
            C3.append(padcol(g["cand_c3i"].values[sel]))
            BI.append(padcol(g["cand_bi"].values[sel]))
            II.append(padcol(g["ii"].values[sel]))
            LAB.append(padcol(lab[sel].astype(np.float32)))
            MASK.append(np.concatenate([np.ones(m),np.zeros(pad)]))
            RIDX.append(padcol(idx[sel].astype(np.int64),fill=-1))
            ss=np.zeros(L,dtype=np.int64); v=short_seq.get(sid)
            if v is not None and len(v): ss[:min(len(v),L)]=v[:L]
            ls=np.zeros(L,dtype=np.int64); v=long_seq.get(sid)
            if v is not None and len(v): ls[:min(len(v),L)]=v[:L]
            SS.append(ss); LS.append(ls)
        T=lambda a,dt: torch.tensor(np.array(a),dtype=dt)
        return dict(sid=sids,
                    num=T(NUM,torch.float32),c3=T(C3,torch.long),bi=T(BI,torch.long),
                    ii=T(II,torch.long),lab=T(LAB,torch.float32),mask=T(MASK,torch.float32),
                    ss=T(SS,torch.long),ls=T(LS,torch.long),ridx=T(RIDX,torch.long))
    tr=pack("train",S_TRAIN); va=pack("val",S_EVAL); te=pack("test",S_EVAL)
    print(f"packed train{tr['num'].shape} val{va['num'].shape} test{te['num'].shape} ({time.time()-t0:.0f}s)", flush=True)

    class DHAR(nn.Module):
        def __init__(s,nc3,nb,d=32,nf=Fn):
            super().__init__()
            s.ec3=nn.Embedding(nc3,d,padding_idx=0); s.eb=nn.Embedding(nb,d,padding_idx=0)
            s.ps=nn.Linear(d,d); s.pl=nn.Linear(d,d)
            s.head=nn.Sequential(nn.Linear(4*d+nf,128),nn.ReLU(),nn.Dropout(0.2),
                                 nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,1))
        def attend(s,q,seq_emb,seqmask,proj):
            # q:[B,S,d] seq_emb:[B,Lh,d] seqmask:[B,Lh]
            qp=proj(q)                                   # [B,S,d]
            sc=torch.einsum("bsd,bld->bsl",qp,seq_emb)/math.sqrt(seq_emb.size(-1))
            sc=sc.masked_fill(~seqmask.bool().unsqueeze(1),-1e9)
            w=torch.softmax(sc,dim=-1); w=torch.nan_to_num(w)
            ctx=torch.einsum("bsl,bld->bsd",w,seq_emb) # [B,S,d]
            return ctx
        def forward(s,num,c3,bi,ss,ls):
            qc=s.ec3(c3); qb=s.eb(bi)                   # [B,S,d]
            se=s.ec3(ss); le=s.ec3(ls)                  # [B,Lh,d]
            sm=(ss>0).float(); lm=(ls>0).float()
            sctx=s.attend(qc,se,sm,s.ps); lctx=s.attend(qc,le,lm,s.pl)
            x=torch.cat([qc,qb,sctx,lctx,num],dim=-1)
            return s.head(x).squeeze(-1)                 # [B,S]

    def listwise(scores,lab,mask):
        # softmax CE with ordinal gains over valid slate positions
        g=torch.tensor(LG,device=scores.device)[lab.long().clamp(0,3)]
        g=g.masked_fill(mask==0,0.0)
        tgt=g/ (g.sum(1,keepdim=True)+1e-9)
        sc=scores.masked_fill(mask==0,-1e9)
        logp=F.log_softmax(sc,dim=1)
        hasp=(g.sum(1)>0).float()
        return -((tgt*logp).sum(1)*hasp).sum()/(hasp.sum()+1e-9)

    def eval_pack(model,pk):
        model.eval(); sc_all=np.zeros(len(feat))
        with torch.no_grad():
            for b in range(0,len(pk["sid"]),256):
                sl=slice(b,b+256)
                sc=model(pk["num"][sl].to(DEV),pk["c3"][sl].to(DEV),pk["bi"][sl].to(DEV),
                         pk["ss"][sl].to(DEV),pk["ls"][sl].to(DEV)).cpu().numpy()
                ridx=pk["ridx"][sl].numpy(); mask=pk["mask"][sl].numpy()
                for i in range(sc.shape[0]):
                    valid=mask[i]>0
                    sc_all[ridx[i][valid]]=sc[i][valid]
        return sc_all

    model=DHAR(NC3,NB).to(DEV); opt=torch.optim.Adam(model.parameters(),lr=2e-3,weight_decay=1e-6)
    print("Training DHAR ...", flush=True)
    best=-1; best_state=None; bad=0
    for ep in range(1,21):
        model.train(); perm=np.random.permutation(len(tr["sid"])); tot=0;nb=0
        for b in range(0,len(perm),256):
            ix=perm[b:b+256]
            sc=model(tr["num"][ix].to(DEV),tr["c3"][ix].to(DEV),tr["bi"][ix].to(DEV),
                     tr["ss"][ix].to(DEV),tr["ls"][ix].to(DEV))
            loss=listwise(sc,tr["lab"][ix].to(DEV),tr["mask"][ix].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step(); tot+=float(loss);nb+=1
        vs=eval_pack(model,va); vd=feat[feat.split=="val"].copy(); vd["s"]=vs[vd.index.values]
        m,_=evaluate(vd,"s"); nd=m["NDCG@10"]
        print(f"  epoch {ep:2d} | loss {tot/nb:.4f} | val NDCG@10 {nd:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if nd>best: best=nd; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=3: print("  early stop",flush=True); break
    model.load_state_dict(best_state)
    ts=eval_pack(model,te); td=feat[feat.split=="test"].copy(); td["dhar_score"]=ts[td.index.values]
    m,n=evaluate(td,"dhar_score")
    print(f"\nDHAR TEST | NDCG@10 {m['NDCG@10']:.4f} | MRR@200 {m['MRR@200']:.4f} | P@1 {m['P@1']:.4f} | HR@10 {m['HR@10']:.4f} (n={n})", flush=True)
    # TRAIN performance (overfitting check)
    tre=pack("train",S_EVAL)
    trs=eval_pack(model,tre); trd2=feat[feat.split=="train"].copy(); trd2["dh"]=trs[trd2.index.values]
    m_tr,_=evaluate(trd2,"dh")
    print(f"DHAR TRAIN | NDCG@10 {m_tr['NDCG@10']:.4f} (test {m['NDCG@10']:.4f})", flush=True)

    # ---- NCF baseline (vectorised) ----
    print("Training NCF ...", flush=True)
    class NCF(nn.Module):
        def __init__(s,nu,ni,d=32):
            super().__init__(); s.ug=nn.Embedding(nu,d); s.ig=nn.Embedding(ni,d)
            s.um=nn.Embedding(nu,d); s.im=nn.Embedding(ni,d)
            s.mlp=nn.Sequential(nn.Linear(2*d,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU()); s.out=nn.Linear(d+32,1)
        def forward(s,u,i):
            return s.out(torch.cat([s.ug(u)*s.ig(i),s.mlp(torch.cat([s.um(u),s.im(i)],-1))],-1)).squeeze(-1)
    ncf=NCF(NU,NI).to(DEV); opt2=torch.optim.Adam(ncf.parameters(),lr=2e-3)
    trd=feat[feat.split=="train"]
    U=torch.tensor(trd["session_id"].values); I=torch.tensor(trd["ii"].values)
    Yb=torch.tensor((trd["label"].values>0).astype(np.float32))
    for ep in range(1,6):
        ncf.train(); perm=torch.randperm(len(U)); tot=0;nb=0
        for b in range(0,len(perm),8192):
            ix=perm[b:b+8192]
            sc=ncf(U[ix].to(DEV),I[ix].to(DEV)); loss=F.binary_cross_entropy_with_logits(sc,Yb[ix].to(DEV))
            opt2.zero_grad(); loss.backward(); opt2.step(); tot+=float(loss);nb+=1
        print(f"  ncf epoch {ep} loss {tot/nb:.4f}", flush=True)
    ncf.eval();
    with torch.no_grad():
        Ut=torch.tensor(td["session_id"].values).to(DEV); It=torch.tensor(td["ii"].values).to(DEV)
        td["ncf_score"]=ncf(Ut,It).cpu().numpy()
    mn,_=evaluate(td,"ncf_score")
    print(f"NCF TEST | NDCG@10 {mn['NDCG@10']:.4f} | MRR@200 {mn['MRR@200']:.4f} | P@1 {mn['P@1']:.4f} | HR@10 {mn['HR@10']:.4f}", flush=True)

    json.dump({"DHAR(ours)":m,"DHAR(ours)_train":m_tr,"NCF(neural)":mn}, open(RES/"results_neural.json","w"), indent=2)
    td[["session_id","cand_wid","label","dhar_score","ncf_score"]].to_parquet(OUT/"test_pred_neural.parquet",index=False)
    torch.save(model.state_dict(), OUT/"dhar.pt")
    print(f"DONE step5 ({time.time()-t0:.0f}s)", flush=True)

if __name__=="__main__":
    main()
