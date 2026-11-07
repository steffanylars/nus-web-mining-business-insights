"""Interpret the trained DHAR model by extracting its two attention towers'
weights on the TEST split (task A5).

Standalone analysis module. Does NOT modify step5_neural.py; it imports DHAR and
the preprocessing utilities and replays encode_frames/pack selection so that each
selected history position can be tagged with its elapsed_days and action type.

Run (from repo root, in the jd_env venv):
    JD_RUN_DIR="/abs/.../local_run/outputs" \
    PYTHONPATH="Final Project/src" python "Final Project/src/dhar_attention.py"
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import OUT, RES
import step5_neural
from step5_neural import brand_bucket
from data_utils import short_mask, select_sequence

AGE_BINS = [(0.0, 1.0, '<1d'), (1.0, 7.0, '1-7d'), (7.0, 30.0, '7-30d'),
            (30.0, 90.0, '30-90d'), (90.0, float('inf'), '>90d')]


def age_bucket(days):
    for lo, hi, name in AGE_BINS:
        if lo <= days < hi:
            return name
    return '>90d'


def attend_weights(model, q, seq, projection):
    """Attention weights [B, width, L] using the model's own math (no weighted sum).

    Mirrors DHAR.attend exactly but returns the renormalised, masked softmax
    weights instead of einsum('bsl,bld->bsd', weight, emb).
    """
    emb = model.ec3(seq)                       # [B, L, d]
    mask = seq > 0                             # [B, L]
    score = torch.einsum('bsd,bld->bsl', projection(q), emb) / math.sqrt(emb.size(-1))
    score = score.masked_fill(~mask[:, None, :], -1e9)
    weight = torch.softmax(score, dim=-1) * mask[:, None, :]
    weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-9)
    return weight                              # [B, width, L]


def build_tagged_sequences(hist, meta, prep):
    """Replay encode_frames selection but also record (elapsed_days, type) per kept
    position. Returns dict tower -> {session_id: (cate_ids[L_sel], days[L_sel],
    types[L_sel])} using the SAME select_sequence choice over surviving indices.
    """
    cat = meta.set_index('wid').cate_id_3.to_dict()
    h = hist.sort_values(['session_id', 'pos']).copy()
    h['_c3'] = h.wid.map(cat).map(prep['category_ids']).fillna(0).astype(int)
    length = prep['history_length']

    out = {}
    for name, subset in [('short', h[short_mask(h)]), ('order', h[h.type.eq('ORD')])]:
        tower = {}
        for sid, g in subset.groupby('session_id', sort=False):
            values = g['_c3'].to_numpy()
            days = g['elapsed_days'].to_numpy()
            types = g['type'].to_numpy()
            keep = values > 0                    # unknown metadata excluded first
            v = values[keep]; d = days[keep]; t = types[keep]
            if len(v) == 0:
                continue
            # select_sequence chooses evenly spaced positions over the surviving set.
            idx = select_sequence(np.arange(len(v)), length)
            tower[sid] = (v[idx], d[idx], t[idx])
        out[name] = tower
    return out


def main():
    dev = torch.device('cpu')
    ckpt = torch.load(OUT / 'dual_seed42.pt', map_location='cpu')
    spec = ckpt['spec']
    model = step5_neural.DHAR(**spec)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    prep = json.loads((OUT / 'neural_preprocessing.json').read_text())
    L = prep['history_length']

    feat = pd.read_parquet(OUT / 'features.parquet')
    hist = pd.read_parquet(OUT / 'history.parquet')
    meta = pd.read_parquet(OUT / 'meta_subset.parquet', columns=['wid', 'cate_id_3', 'brand_id'])

    # ---- reconstruct tagged sequences and assert vs. encode_frames ----
    tagged = build_tagged_sequences(hist, meta, prep)
    ef_frame, ef_seqs, _ = step5_neural.encode_frames(feat, hist, meta, prep)
    checked = 0
    for name in ('short', 'order'):
        for sid in list(ef_seqs[name].keys())[:200]:
            ref = np.asarray(ef_seqs[name][sid])
            if len(ref) == 0:
                continue
            got = tagged[name].get(sid, (np.array([]),))[0]
            assert np.array_equal(ref, got), f'mismatch {name} session {sid}: {ref} vs {got}'
            checked += 1
            if checked >= 50:
                break
        if checked >= 50:
            break
    print(f'[check] reconstructed cate-id sequences match encode_frames on {checked} sessions')

    # ---- candidate c3 per test session, sorted by cand_wid as pack() does ----
    cat_map = meta.set_index('wid').cate_id_3.to_dict()
    brand_map = meta.set_index('wid').brand_id.to_dict()
    test = feat[feat.split.eq('test')].copy()
    test['_c3'] = test.cand_wid.map(cat_map).map(prep['category_ids']).fillna(0).astype(int)
    test['_bi'] = test.cand_wid.map(brand_map).map(brand_bucket).astype(int)

    test_sids = sorted(test.session_id.unique())

    # Accumulators: per tower, summed attention mass per bucket, and pair count.
    agg = {t: {'age': {b[2]: 0.0 for b in AGE_BINS}, 'type': {}, 'n': 0}
           for t in ('short', 'order')}

    example = {'sid': None, 'ncand': -1}  # pick longest-history test session w/ positive

    with torch.no_grad():
        for sid in test_sids:
            g = test[test.session_id.eq(sid)].sort_values('cand_wid')
            cand_c3 = torch.tensor(g['_c3'].to_numpy(), dtype=torch.long)[None, :]  # [1,W]
            cand_mask = cand_c3[0] > 0
            qc = model.ec3(cand_c3)                                 # [1, W, d]

            for tower, projection in (('short', model.ps), ('order', model.pl)):
                rec = tagged[tower].get(sid)
                if rec is None:
                    continue
                v, days, types = rec
                seq = torch.zeros((1, L), dtype=torch.long)
                seq[0, :len(v)] = torch.tensor(v, dtype=torch.long)
                w = attend_weights(model, qc, seq, projection)[0]    # [W, L]
                w = w[:, :len(v)]                                    # drop padding cols
                # Only real candidates (c3>0) are valid queries.
                w_valid = w[cand_mask]                               # [n_valid, len(v)]
                if w_valid.numel() == 0:
                    continue
                wnp = w_valid.numpy()                                # rows sum ~1
                agebins = np.array([age_bucket(dd) for dd in days])
                for lo, hi, bname in AGE_BINS:
                    sel = agebins == bname
                    if sel.any():
                        agg[tower]['age'][bname] += float(wnp[:, sel].sum())
                for tp in np.unique(types):
                    sel = types == tp
                    agg[tower]['type'].setdefault(str(tp), 0.0)
                    agg[tower]['type'][str(tp)] += float(wnp[:, sel].sum())
                agg[tower]['n'] += wnp.shape[0]

            # track example: longest history with a labelled positive candidate
            if (g.label.to_numpy() > 0).any():
                hlen = (len(tagged['short'].get(sid, ([],))[0]) +
                        len(tagged['order'].get(sid, ([],))[0]))
                if hlen > example['ncand']:
                    example = {'sid': sid, 'ncand': hlen}

    # ---- mean attention mass = summed mass / number of (query,candidate) pairs ----
    rows = []
    for tower in ('short', 'order'):
        n = max(agg[tower]['n'], 1)
        for lo, hi, bname in AGE_BINS:
            rows.append(dict(tower=tower, dimension='age_bin', bucket=bname,
                             mean_attention_mass=agg[tower]['age'][bname] / n))
        for tp, mass in sorted(agg[tower]['type'].items()):
            rows.append(dict(tower=tower, dimension='action_type', bucket=tp,
                             mean_attention_mass=mass / n))
    df = pd.DataFrame(rows)
    csv_path = RES / 'dhar_attention_by_age_type.csv'
    df.to_csv(csv_path, index=False)

    # sanity
    for tower in ('short', 'order'):
        age_sum = df[(df.tower == tower) & (df.dimension == 'age_bin')].mean_attention_mass.sum()
        print(f'[sanity] {tower} age-bin mass sum = {age_sum:.4f} (pairs n={agg[tower]["n"]})')

    # ---- figure: grouped bar of mean attention mass by age bin ----
    bin_names = [b[2] for b in AGE_BINS]
    short_vals = [df[(df.tower == 'short') & (df.bucket == b) & (df.dimension == 'age_bin')]
                  .mean_attention_mass.values[0] for b in bin_names]
    order_vals = [df[(df.tower == 'order') & (df.bucket == b) & (df.dimension == 'age_bin')]
                  .mean_attention_mass.values[0] for b in bin_names]
    x = np.arange(len(bin_names)); bw = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - bw / 2, short_vals, bw, label='SHORT tower', color='#2b8cbe')
    ax.bar(x + bw / 2, order_vals, bw, label='ORDER tower', color='#e34a33')
    ax.set_xticks(x); ax.set_xticklabels(bin_names)
    ax.set_xlabel('Event age bin'); ax.set_ylabel('Mean attention mass')
    ax.set_title('DHAR attention mass by event age (TEST split)')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig_path = RES / 'fig_dhar_attention.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    # ---- worked example: top-5 attended events for one positive candidate ----
    sid = example['sid']
    inv_cat = {v: k for k, v in prep['category_ids'].items()}
    g = test[test.session_id.eq(sid)].sort_values('cand_wid').reset_index(drop=True)
    pos_rows = g[(g.label.to_numpy() > 0) & (g['_c3'].to_numpy() > 0)]
    print(f'\n[example] test session {sid}: {len(g)} candidates, '
          f'short_len={len(tagged["short"].get(sid, ([],))[0])}, '
          f'order_len={len(tagged["order"].get(sid, ([],))[0])}')
    if len(pos_rows):
        # position of the chosen positive candidate within the cand-sorted slate
        cand_pos = g.index.get_loc(pos_rows.index[0])
        cand_wid = int(g.loc[cand_pos, 'cand_wid'])
        cand_c3 = torch.tensor(g['_c3'].to_numpy(), dtype=torch.long)[None, :]
        qc = model.ec3(cand_c3)
        attended = []
        with torch.no_grad():
            for tower, projection in (('short', model.ps), ('order', model.pl)):
                rec = tagged[tower].get(sid)
                if rec is None:
                    continue
                v, days, types = rec
                seq = torch.zeros((1, L), dtype=torch.long)
                seq[0, :len(v)] = torch.tensor(v, dtype=torch.long)
                w = attend_weights(model, qc, seq, projection)[0][cand_pos, :len(v)].numpy()
                for i in range(len(v)):
                    attended.append((tower, float(days[i]), str(types[i]),
                                     inv_cat.get(int(v[i]), '?'), float(w[i])))
        attended.sort(key=lambda r: r[4], reverse=True)
        print(f'  candidate cand_wid={cand_wid} (positive), top-5 attended events:')
        print('  rank  tower  age_days   type   cate_id_3     weight')
        for rank, (tw, dd, tp, c3, wt) in enumerate(attended[:5], 1):
            print(f'  {rank:>4}  {tw:<6} {dd:>8.2f}  {tp:<5}  {str(c3):<12} {wt:.4f}')

    print(f'\n[written] {csv_path}')
    print(f'[written] {fig_path}')
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
