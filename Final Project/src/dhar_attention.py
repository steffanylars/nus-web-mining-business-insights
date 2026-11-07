"""Interpret the trained DHAR model by extracting its two attention towers'
weights on the TEST split (task A5).

Standalone analysis module. Does NOT modify step5_neural.py; it imports DHAR and
the preprocessing utilities and replays encode_frames/pack selection so that each
selected history position can be tagged with its elapsed_days and action type.

Vectorised: candidates for all test queries are scored in batches on the same
device logic as step5_neural (CUDA, then MPS, then CPU); there is no per-query
Python loop for the attention weights.

Run (from repo root, in the jd_env venv):
    JD_RUN_DIR="/abs/.../local_run/outputs" \
    PYTHONPATH="Final Project/src" python "Final Project/src/dhar_attention.py"
"""
import json
import math
import time

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import OUT, RES, SEED
import step5_neural
from step5_neural import brand_bucket, device
from data_utils import short_mask, select_sequence

AGE_BINS = [(0.0, 1.0, '<1d'), (1.0, 7.0, '1-7d'), (7.0, 30.0, '7-30d'),
            (30.0, 90.0, '30-90d'), (90.0, float('inf'), '>90d')]
AGE_NAMES = [b[2] for b in AGE_BINS]
AGE_EDGES = np.array([1.0, 7.0, 30.0, 90.0])          # np.digitize -> bin index 0..4
TOWER_TYPES = {'short': ['CLICK', 'CART'], 'order': ['ORD']}
BATCH = 256                                            # test queries scored per batch
MAX_SECONDS = 180                                      # fall back to a sample beyond this


def build_tagged_sequences(hist, meta, prep):
    """Replay encode_frames selection but also record (cate_id, elapsed_days, type)
    per kept position. Returns tower -> {sid: (cate_ids, days, types)} using the SAME
    select_sequence choice over surviving (known-category) indices.
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
            keep = values > 0                          # unknown metadata excluded first
            v = values[keep]
            if len(v) == 0:
                continue
            idx = select_sequence(np.arange(len(v)), length)
            tower[sid] = (v[idx], g['elapsed_days'].to_numpy()[keep][idx],
                          g['type'].to_numpy()[keep][idx])
        out[name] = tower
    return out


def _tower_mass_proj(model, dev, sids, tagged_tower, cand_c3_by_sid, L, types, projection):
    n_types = len(types)
    type_index = {t: i for i, t in enumerate(types)}
    sids = [s for s in sids if s in tagged_tower]
    seq = np.zeros((len(sids), L), dtype=np.int64)
    age_oh = np.zeros((len(sids), L, len(AGE_NAMES)), dtype=np.float32)
    type_oh = np.zeros((len(sids), L, n_types), dtype=np.float32)
    for j, sid in enumerate(sids):
        v, days, tp = tagged_tower[sid]
        m = len(v)
        seq[j, :m] = v
        abin = np.digitize(days, AGE_EDGES)
        age_oh[j, np.arange(m), abin] = 1.0
        for p in range(m):
            type_oh[j, p, type_index[str(tp[p])]] = 1.0

    age_mass = np.zeros(len(AGE_NAMES), dtype=np.float64)
    type_mass = np.zeros(n_types, dtype=np.float64)
    # Uniform-attention baseline (what 1/len weights over the eligible events would give):
    base_age = np.zeros(len(AGE_NAMES), dtype=np.float64)
    base_type = np.zeros(n_types, dtype=np.float64)
    n_pairs = 0
    sqrt_d = math.sqrt(model.ec3.embedding_dim)
    for a in range(0, len(sids), BATCH):
        chunk = sids[a:a + BATCH]
        b = len(chunk)
        wmax = max(len(cand_c3_by_sid[s]) for s in chunk)
        cc = np.zeros((b, wmax), dtype=np.int64)
        cmask = np.zeros((b, wmax), dtype=np.float32)
        for i, sid in enumerate(chunk):
            cv = cand_c3_by_sid[sid]
            cc[i, :len(cv)] = cv
            cmask[i, :len(cv)] = 1.0
            n_pairs += len(cv)
        sq = torch.from_numpy(seq[a:a + b]).to(dev)
        qc = model.ec3(torch.from_numpy(cc).to(dev))                # [b, W, d]
        emb = model.ec3(sq)                                         # [b, L, d]
        smask = (sq > 0)
        score = torch.einsum('bwd,bld->bwl', projection(qc), emb) / sqrt_d
        score = score.masked_fill(~smask[:, None, :], -1e9)
        w = torch.softmax(score, dim=-1) * smask[:, None, :].float()
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)             # [b, W, L]
        cm = torch.from_numpy(cmask).to(dev)
        w = w * cm[:, :, None]                                      # zero padded candidates
        # Uniform baseline: equal weight 1/len over a session's eligible positions.
        unif = smask.float() / smask.float().sum(-1, keepdim=True).clamp_min(1)   # [b, L]
        wu = unif[:, None, :] * cm[:, :, None]                      # [b, W, L]
        age = torch.from_numpy(age_oh[a:a + b]).to(dev)            # [b, L, 5]
        typ = torch.from_numpy(type_oh[a:a + b]).to(dev)          # [b, L, T]
        age_mass += torch.einsum('bwl,blk->k', w, age).cpu().double().numpy()
        type_mass += torch.einsum('bwl,blt->t', w, typ).cpu().double().numpy()
        base_age += torch.einsum('bwl,blk->k', wu, age).cpu().double().numpy()
        base_type += torch.einsum('bwl,blt->t', wu, typ).cpu().double().numpy()
    return age_mass, type_mass, base_age, base_type, n_pairs


def main():
    dev = device()
    print('device', dev, flush=True)
    t0 = time.time()
    ckpt = torch.load(OUT / 'dual_seed42.pt', map_location=dev)
    spec = ckpt['spec']
    model = step5_neural.DHAR(**spec).to(dev)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    prep = json.loads((OUT / 'neural_preprocessing.json').read_text())
    L = prep['history_length']

    feat = pd.read_parquet(OUT / 'features.parquet')
    hist = pd.read_parquet(OUT / 'history.parquet')
    meta = pd.read_parquet(OUT / 'meta_subset.parquet', columns=['wid', 'cate_id_3', 'brand_id'])

    test = feat[feat.split.eq('test')].copy()
    test_sids = sorted(test.session_id.unique())
    n_test = len(test_sids)

    # Optional fixed sample only if the full run would be too slow (stated in outputs).
    sample_note = f'all {n_test} test queries'

    # Tag history positions for test sessions only (keeps the groupby small).
    hist_test = hist[hist.session_id.isin(set(test_sids))]
    tagged = build_tagged_sequences(hist_test, meta, prep)

    # Self-check: reconstructed cate-id sequences equal encode_frames' selection.
    _, ef_seqs, _ = step5_neural.encode_frames(feat, hist, meta, prep)
    checked = 0
    for name in ('short', 'order'):
        for sid in list(ef_seqs[name].keys()):
            ref = np.asarray(ef_seqs[name][sid])
            if len(ref) == 0 or sid not in tagged[name]:
                continue
            assert np.array_equal(ref, tagged[name][sid][0]), f'mismatch {name} {sid}'
            checked += 1
            if checked >= 50:
                break
        if checked >= 50:
            break
    print(f'[check] reconstructed sequences match encode_frames on {checked} sessions')

    # Candidate category per test session (valid c3>0 only; order within a session
    # does not affect the summed mass). Sorted by cand_wid as pack() does.
    cat_map = meta.set_index('wid').cate_id_3.to_dict()
    test = test.sort_values(['session_id', 'cand_wid'])
    test['_c3'] = test.cand_wid.map(cat_map).map(prep['category_ids']).fillna(0).astype(int)
    cand_c3_by_sid = {sid: g['_c3'].to_numpy()[g['_c3'].to_numpy() > 0]
                      for sid, g in test.groupby('session_id', sort=True)}

    def run_towers(sids):
        res = {}
        with torch.no_grad():
            for tower, proj in (('short', model.ps), ('order', model.pl)):
                res[tower] = _tower_mass_proj(model, dev, sids, tagged[tower],
                                              cand_c3_by_sid, L, TOWER_TYPES[tower], proj)
        return res

    res = run_towers(test_sids)
    if time.time() - t0 > MAX_SECONDS:
        rng = np.random.default_rng(SEED)
        sample = sorted(rng.choice(test_sids, size=min(500, n_test), replace=False).tolist())
        sample_note = f'fixed sample of {len(sample)} test queries (seed {SEED})'
        print(f'[slow] full run exceeded {MAX_SECONDS}s; using {sample_note}', flush=True)
        res = run_towers(sample)

    def _row(tower, dim, bucket, mass, base, n):
        share = base / n
        return dict(tower=tower, dimension=dim, bucket=bucket,
                    mean_attention_mass=mass / n, baseline_share=share,
                    ratio_attn_over_baseline=(mass / base if base else float('nan')),
                    queries=sample_note)

    rows = []
    for tower in ('short', 'order'):
        age_mass, type_mass, base_age, base_type, n = res[tower]
        n = max(n, 1)
        for k, bname in enumerate(AGE_NAMES):
            rows.append(_row(tower, 'age_bin', bname, age_mass[k], base_age[k], n))
        for i, tp in enumerate(TOWER_TYPES[tower]):
            rows.append(_row(tower, 'action_type', tp, type_mass[i], base_type[i], n))
    df = pd.DataFrame(rows)
    csv_path = RES / 'dhar_attention_by_age_type.csv'
    df.to_csv(csv_path, index=False)

    for tower in ('short', 'order'):
        s = df[(df.tower == tower) & (df.dimension == 'age_bin')].mean_attention_mass.sum()
        print(f'[sanity] {tower} age-bin mass sum = {s:.4f}')
    print('[note] the SHORT tower has 0 attention mass beyond 90 days BY CONSTRUCTION: its '
          'sequence only holds clicks <=7d and carts <=90d. baseline_share is the uniform-attention '
          'mass (share of eligible events in each bin); ratio_attn_over_baseline > 1 means the tower '
          'over-weights that bin vs uniform.')

    # Figure: grouped bar of mean attention mass by age bin.
    sv = [df[(df.tower == 'short') & (df.bucket == b) & (df.dimension == 'age_bin')]
          .mean_attention_mass.values[0] for b in AGE_NAMES]
    ov = [df[(df.tower == 'order') & (df.bucket == b) & (df.dimension == 'age_bin')]
          .mean_attention_mass.values[0] for b in AGE_NAMES]
    x = np.arange(len(AGE_NAMES)); bw = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - bw / 2, sv, bw, label='SHORT tower', color='#2b8cbe')
    ax.bar(x + bw / 2, ov, bw, label='ORDER tower', color='#e34a33')
    ax.set_xticks(x); ax.set_xticklabels(AGE_NAMES)
    ax.set_xlabel('Event age bin'); ax.set_ylabel('Mean attention mass')
    ax.set_title(f'DHAR attention mass by event age (TEST, {sample_note})')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig_path = RES / 'fig_dhar_attention.png'
    fig.savefig(fig_path, dpi=150); plt.close(fig)

    # Worked example: top-5 attended events for one long-history positive candidate.
    inv_cat = {v: k for k, v in prep['category_ids'].items()}
    best = max(test_sids, key=lambda s: len(tagged['short'].get(s, ([],))[0]) +
               len(tagged['order'].get(s, ([],))[0]))
    g = test[test.session_id.eq(best)].sort_values('cand_wid').reset_index(drop=True)
    pos = g[(g.label.to_numpy() > 0) & (g['_c3'].to_numpy() > 0)]
    print(f'\n[example] test session {best}: {len(g)} candidates, '
          f"short_len={len(tagged['short'].get(best, ([],))[0])}, "
          f"order_len={len(tagged['order'].get(best, ([],))[0])}")
    if len(pos):
        cand_pos = g.index.get_loc(pos.index[0])
        attended = []
        with torch.no_grad():
            qc = model.ec3(torch.tensor(g['_c3'].to_numpy(), dtype=torch.long, device=dev)[None, :])
            for tower, proj in (('short', model.ps), ('order', model.pl)):
                rec = tagged[tower].get(best)
                if rec is None:
                    continue
                v, days, tps = rec
                sq = torch.zeros((1, L), dtype=torch.long, device=dev)
                sq[0, :len(v)] = torch.tensor(v, dtype=torch.long, device=dev)
                emb = model.ec3(sq); smask = sq > 0
                sc = torch.einsum('bwd,bld->bwl', proj(qc), emb) / math.sqrt(model.ec3.embedding_dim)
                sc = sc.masked_fill(~smask[:, None, :], -1e9)
                w = torch.softmax(sc, dim=-1) * smask[:, None, :].float()
                w = (w / w.sum(-1, keepdim=True).clamp_min(1e-9))[0, cand_pos, :len(v)].cpu().numpy()
                for i in range(len(v)):
                    attended.append((tower, float(days[i]), str(tps[i]),
                                     inv_cat.get(int(v[i]), '?'), float(w[i])))
        attended.sort(key=lambda r: r[4], reverse=True)
        print(f'  candidate cand_wid={int(g.loc[cand_pos, "cand_wid"])} (positive), top-5 attended:')
        print('  rank  tower  age_days   type   cate_id_3     weight')
        for rank, (tw, dd, tp, c3, wt) in enumerate(attended[:5], 1):
            print(f'  {rank:>4}  {tw:<6} {dd:>8.2f}  {tp:<5}  {str(c3):<12} {wt:.4f}')

    print(f'\n[time] {time.time() - t0:.1f}s  [queries] {sample_note}')
    print(f'[written] {csv_path}\n[written] {fig_path}')
    print(df.drop(columns=['queries']).to_string(index=False))


if __name__ == '__main__':
    main()
