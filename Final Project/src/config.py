"""Shared portable configuration; new runs do not overwrite draft artifacts."""
import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get('JD_RUN_DIR', BASE / 'artifacts' / 'corrected')).resolve()
OUT = RES / 'model'
FIG = RES / 'figures'
SRC_USER = Path(os.environ.get('JD_USER_TXT', BASE / 'data' / 'user_behavior_data.txt'))
SRC_PRODUCT = Path(os.environ.get('JD_PRODUCT_TXT', BASE / 'data' / 'product_meta_data.txt'))
SEED = int(os.environ.get('JD_SEED', '42'))
HISTORY_LENGTH = int(os.environ.get('JD_HISTORY_LENGTH', '20'))
EPOCHS = int(os.environ.get('JD_EPOCHS', '20'))
BATCH_SIZE = int(os.environ.get('JD_BATCH_SIZE', '128'))
THREADS = int(os.environ.get('JD_THREADS', '6'))
SCHEMA_VERSION = 2

def prepare():
    for path in (RES, OUT, FIG):
        path.mkdir(parents=True, exist_ok=True)

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

def feature_columns(frame):
    return [c for c in frame.columns if c not in
            {'session_id', 'source_row_id', 'cand_wid', 'label', 'split'}]
