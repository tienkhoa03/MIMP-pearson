import os
from typing import List, Tuple, Dict, Optional, Callable

import pandas as pd


def find_csv_file(folder: str, dataset: str, case_keywords: List[List[str]], stream_str: str) -> Optional[str]:
    for fname in os.listdir(folder):
        if not fname.endswith('.csv'):
            continue
        if dataset not in fname:
            continue
        if stream_str not in fname:
            continue
        low = fname.lower()
        ok = True
        for kw_group in case_keywords:
            # kw_group is a list of alternative substrings; at least one must appear
            if not any(kw in low for kw in kw_group if kw):
                ok = False
                break
        if ok:
            return os.path.join(folder, fname)
    return None


def read_metrics_from_file(path: str, reducer: Callable = None) -> Tuple[float, float]:
    df = pd.read_csv(path, index_col=0)
    mre = float('nan')
    t = float('nan')
    if reducer is None:
        reducer = pd.Series.mean
    if 'mape' in df.columns:
        mre = reducer(df['mape']) * 100.0
    if 'tot_time' in df.columns:
        t = reducer(df['tot_time'])
    return mre, t


def collect_dataset_metrics(folder: str, dataset: str, streams: List[float]) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    # Map cases to deterministic filename filters and aggregation rules.
    # P and D use max, M and DM use min.
    cases = {
        'DM': {'match': lambda name: ('data+state' in name and 'transfer' in name), 'reducer': min},
        'P': {'match': lambda name: ('alone' in name), 'reducer': max},
        'D': {'match': lambda name: ('_data_' in name or '_data' in name) and 'data+state' not in name, 'reducer': max},
        'M': {'match': lambda name: ('_transfer_' in name or '_transfer' in name) and 'data+state' not in name, 'reducer': min},
    }

    results = {c: {} for c in cases}
    for s in streams:
        stream_str = str(s)
        for label, rule in cases.items():
            path = None
            for fname in os.listdir(folder):
                if not fname.endswith('.csv'):
                    continue
                if dataset not in fname:
                    continue
                if stream_str not in fname:
                    continue
                low = fname.lower()
                if rule['match'](low):
                    path = os.path.join(folder, fname)
                    break
            if path:
                mre, t = read_metrics_from_file(path, reducer=rule['reducer'])
            else:
                mre, t = float('nan'), float('nan')
            results[label][stream_str] = {'mre': mre, 'time': t, 'file': path}
    return results
