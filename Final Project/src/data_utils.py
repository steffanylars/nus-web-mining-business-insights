"""JDSearch field semantics shared by training and inference."""
import numpy as np
import pandas as pd

def split_list(value):
    return str(value).split('_') if value is not None and str(value).strip() else []

def tok(value):
    if value is None or pd.isna(value) or str(value).strip() in {'', '-1'}:
        return set()
    return {part for part in str(value).split('\x18') if part}

def elapsed_days(gaps, n_events):
    values = np.asarray(gaps, dtype=np.float64)
    if len(values) != n_events + 1 or not np.isfinite(values).all() or (values < 0).any() or values[0] != 0:
        raise ValueError('History requires n+1 finite nonnegative gaps with initial zero')
    return np.cumsum(values[::-1])[::-1][1:] / 86400.0

def short_mask(history):
    """Intent channel: clicks <=7 days or carts <=90 days."""
    return ((history['type'].eq('CLICK') & history.elapsed_days.le(7)) |
            (history['type'].eq('CART') & history.elapsed_days.le(90)))

def select_sequence(values, length):
    """Evenly sample the complete chronological horizon, keeping both endpoints."""
    if length < 1:
        raise ValueError('Sequence length must be positive')
    values = np.asarray(values)
    if len(values) <= length:
        return values
    return values[np.linspace(0, len(values) - 1, length, dtype=int)]
