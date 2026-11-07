"""Seed-sensitivity check for DHAR vs the MLP control (standalone; not part of the step pipeline).

Trains the dual-tower DHAR and the no-tower MLP control on seeds 43 and 44 and combines them with
the existing seed-42 run, so the seed-to-seed spread can be compared against the DHAR-minus-MLP gap.
All seeds use the same tables and machine. Seed 42 is read from results_neural.json and never retrained.

Run against the committed re-run outputs (point JD_RUN_DIR at the folder holding the Step 2/3 tables,
the seed-42 checkpoints and results_neural.json):

    JD_RUN_DIR="/path/to/repo/local_run/outputs" python "Final Project/src/multiseed.py"
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

# src/ must be importable; JD_RUN_DIR must be set before importing config.
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
if 'JD_RUN_DIR' not in os.environ:
    raise SystemExit('Set JD_RUN_DIR to the run folder with the tables, seed-42 checkpoints and '
                     'results_neural.json, e.g. <repo>/local_run/outputs')

from config import OUT, RES, HISTORY_LENGTH, prepare  # noqa: E402
from eval_utils import evaluate  # noqa: E402
from step5_neural import (fit_preprocessing, encode_frames, pack, train_one,  # noqa: E402
                          predict_pack, device, NAMES)

SEEDS_NEW = [43, 44]
MODELS = {'dual': 'DHAR', 'mlp': 'MLP'}  # variant -> short label


def _read_seed42(results_path):
    data = json.load(open(results_path))
    out = {}
    for variant, short in MODELS.items():
        name = NAMES[variant]
        out[short] = dict(seed=42,
                          val_ndcg10=data[f'{name}_val']['NDCG@10'],
                          test_ndcg10=data[name]['NDCG@10'])
    return out


def main():
    prepare()
    dev = device()
    print('run dir', RES, '| device', dev, flush=True)
    feat = pd.read_parquet(OUT / 'features.parquet')
    hist = pd.read_parquet(OUT / 'history.parquet')
    meta = pd.read_parquet(OUT / 'meta_subset.parquet')
    prep = fit_preprocessing(feat, hist, meta, HISTORY_LENGTH)
    frame, sequences, _ = encode_frames(feat, hist, meta, prep)
    packs = {s: pack(frame, sequences, prep, s, 64 if s == 'train' else 200)
             for s in ['train', 'val', 'test']}

    rows = []
    seed42 = _read_seed42(RES / 'results_neural.json')  # reuse, never retrain
    for short, rec in seed42.items():
        rows.append(dict(model=short, **rec))

    for seed in SEEDS_NEW:
        for variant, short in MODELS.items():
            model = train_one(variant, prep, frame, packs, seed, dev)
            val = frame[frame.split.eq('val')].copy()
            val['score'] = predict_pack(model, packs['val'], len(frame), dev)[val.index.to_numpy()]
            test = frame[frame.split.eq('test')].copy()
            test['score'] = predict_pack(model, packs['test'], len(frame), dev)[test.index.to_numpy()]
            rows.append(dict(model=short, seed=seed,
                             val_ndcg10=evaluate(val, 'score')[0]['NDCG@10'],
                             test_ndcg10=evaluate(test, 'score')[0]['NDCG@10']))
            del model

    summary = pd.DataFrame(rows).sort_values(['model', 'seed']).reset_index(drop=True)
    out_csv = RES / 'multiseed_summary.csv'
    summary.to_csv(out_csv, index=False)
    print('\nwrote', out_csv)
    print(summary.to_string(index=False))

    print('\n=== per-model mean (sd) over seeds 42/43/44 ===')
    for short in ['DHAR', 'MLP']:
        sub = summary[summary.model == short]
        print(f'{short}: test {sub.test_ndcg10.mean():.5f} (sd {sub.test_ndcg10.std(ddof=1):.5f}); '
              f'val {sub.val_ndcg10.mean():.5f} (sd {sub.val_ndcg10.std(ddof=1):.5f})')

    print('\n=== DHAR - MLP test NDCG@10 per seed ===')
    piv = summary.pivot(index='seed', columns='model', values='test_ndcg10')
    for seed in piv.index:
        print(f'seed {seed}: {piv.loc[seed, "DHAR"] - piv.loc[seed, "MLP"]:+.5f}')


if __name__ == '__main__':
    main()
