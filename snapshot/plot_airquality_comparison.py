"""
Plot MAE vs Pearson coefficient for Airquality experiment results.

Usage:
    python scripts/plot_airquality_comparison.py --dir snapshot/exp_results/Airquality --out plots/airquality.png

The script searches CSV files in the target directory, extracts the pearson value
from filenames (or treats 'no_pearson' as 0), reads the `opt_mae` field from each
CSV, and plots 6 subplots for combinations: DAC/DAMC with evals 0.1, 0.3, 0.5.
"""
import argparse
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd


FILENAME_PEARSON = re.compile(r"_pearson_([0-9]+(?:\.[0-9]+)?)")
FILENAME_EVAL = re.compile(r"eval[_-]?([0-9]+(?:\.[0-9]+)?)")


def extract_info_from_filename(fn):
    base = os.path.basename(fn)
    # pearson: 'no_pearson' -> 0
    if 'no_pearson' in base:
        pearson = 0.0
    else:
        m = FILENAME_PEARSON.search(base)
        pearson = float(m.group(1)) if m else None

    # method: look for DAC or DAMC in filename
    method = 'unknown'
    if 'DAC' in base:
        method = 'DAC'
    if 'DAMC' in base:
        method = 'DAMC'

    # eval value
    m2 = FILENAME_EVAL.search(base)
    eval_v = float(m2.group(1)) if m2 else None

    return method, eval_v, pearson


def read_mae_from_csv(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    # prefer 'opt_mae' column, fallback to common alternatives
    for col in ("opt_mae", "mae", "opt_MAE", "MAE"):
        if col in df.columns:
            try:
                return float(df.iloc[0][col])
            except Exception:
                pass
    # as a last resort, try second column if first is index
    if df.shape[1] >= 2:
        try:
            return float(df.iloc[0, 1])
        except Exception:
            pass
    return None


def gather_results(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith('.csv')]
    results = defaultdict(list)  # key: (method, eval) -> list of (pearson, mae)
    records = []
    for f in files:
        method, eval_v, pearson = extract_info_from_filename(f)
        if method not in ('DAC', 'DAMC'):
            continue
        if eval_v is None:
            continue
        if pearson is None:
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if df.shape[0] == 0:
            continue
        rec = df.iloc[0].to_dict()
        rec.update({'__filename': os.path.basename(f), 'method': method, 'eval': eval_v, 'pearson': pearson})
        records.append(rec)
        mae = None
        for col in ("opt_mae", "mae", "opt_MAE", "MAE"):
            if col in rec:
                try:
                    mae = float(rec[col])
                except Exception:
                    mae = None
                break
        if mae is not None:
            key = (method, eval_v)
            results[key].append((pearson, mae))

    return results, pd.DataFrame(records) if records else pd.DataFrame()


def plot_results(results, out_path=None):
    # desired evals and methods
    evals = [0.1, 0.3, 0.5]
    methods = ['DAC', 'DAMC']

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.reshape(-1, 2)
    idx = 0
    for i, ev in enumerate(evals):
        for j, m in enumerate(methods):
            ax = axes[i, j]
            key = (m, ev)
            data = sorted(results.get(key, []), key=lambda x: x[0])
            if data:
                xs, ys = zip(*data)
                ax.plot(xs, ys, marker='o')
            else:
                xs, ys = [], []
            ax.set_title(f"{m} — eval {ev}")
            ax.set_xlabel('Pearson coefficient')
            ax.set_ylabel('MAE')
            ax.set_xlim(0, 0.9)
            ax.set_xticks([round(x*0.1, 1) for x in range(0, 10)])
            ax.grid(True, linestyle='--', alpha=0.4)
            idx += 1

    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path)
        print(f"Saved figure to {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='exp_results/Airquality2', help='Directory with Airquality CSV results')
    parser.add_argument('--out', default='plots/airquality_comparison.png', help='Output path for combined figure')
    parser.add_argument('--aggregate', default='plots/airquality_aggregated.csv', help='Path to save aggregated CSV of results')
    args = parser.parse_args()
    if not os.path.isdir(args.dir):
        raise SystemExit(f"Directory not found: {args.dir}")

    results, agg_df = gather_results(args.dir)
    if agg_df is None or agg_df.empty:
        raise SystemExit("No matching results found in directory. Check filenames and location.")

    # save aggregated CSV
    os.makedirs(os.path.dirname(args.aggregate), exist_ok=True)
    agg_df.to_csv(args.aggregate, index=False)
    print(f"Saved aggregated CSV to {args.aggregate} ({len(agg_df)} rows)")

    plot_results(results, args.out)


if __name__ == '__main__':
    main()
