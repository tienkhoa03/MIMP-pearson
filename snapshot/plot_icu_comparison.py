"""
Plot MAE vs Pearson coefficient for ICU experiment results.

Usage:
    python scripts/plot_icu_comparison.py --dir snapshot/exp_results/ICU --out plots/icu_comparison.png

The script searches CSV files in the target directory, extracts the pearson value
from filenames (or treats 'no_pearson' as 0), reads the `opt_mae` field from each
CSV, and plots 3x2 subplots for combinations: DAC/DAMC with evals 0.1, 0.3, 0.5.
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
    if 'no_pearson' in base:
        pearson = 0.0
    else:
        m = FILENAME_PEARSON.search(base)
        pearson = float(m.group(1)) if m else None

    method = 'unknown'
    if 'DAC' in base:
        method = 'DAC'
    if 'DAMC' in base:
        method = 'DAMC'

    m2 = FILENAME_EVAL.search(base)
    eval_v = float(m2.group(1)) if m2 else None

    return method, eval_v, pearson


def read_mae_from_csv(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    for col in ("opt_mae", "mae", "opt_MAE", "MAE"):
        if col in df.columns:
            try:
                return float(df.iloc[0][col])
            except Exception:
                pass
    if df.shape[1] >= 2:
        try:
            return float(df.iloc[0, 1])
        except Exception:
            pass
    return None


def gather_results(directory, aggregate_out=None):
    if not os.path.isdir(directory):
        raise FileNotFoundError(directory)
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith('.csv')]
    results = defaultdict(list)
    records = []
    for f in files:
        method, eval_v, pearson = extract_info_from_filename(f)
        if method not in ('DAC', 'DAMC'):
            continue
        if eval_v is None or pearson is None:
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
            results[(method, eval_v)].append((pearson, mae))

    if aggregate_out and records:
        agg_df = pd.DataFrame(records)
        os.makedirs(os.path.dirname(aggregate_out), exist_ok=True)
        agg_df.to_csv(aggregate_out, index=False)
        print(f"Saved aggregated CSV to {aggregate_out}")
    else:
        agg_df = pd.DataFrame(records) if records else pd.DataFrame()

    return results, agg_df


def plot_results(results, out_path=None):
    evals = [0.1, 0.3, 0.5]
    methods = ['DAC', 'DAMC']

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    for i, ev in enumerate(evals):
        for j, m in enumerate(methods):
            ax = axes[i, j]
            data = sorted(results.get((m, ev), []), key=lambda x: x[0])
            if data:
                xs, ys = zip(*data)
                ax.plot(xs, ys, marker='o')
            ax.set_title(f"{m} — eval {ev}")
            ax.set_xlabel('Pearson coefficient')
            ax.set_ylabel('MAE')
            ax.set_xlim(0, 0.9)
            ax.set_xticks([round(x*0.1, 1) for x in range(0, 10)])
            ax.grid(True, linestyle='--', alpha=0.4)

    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path)
        print(f"Saved figure to {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='exp_results/ICU', help='Directory with ICU CSV results')
    parser.add_argument('--out', default='plots/icu_comparison.png', help='Output path for combined figure')
    parser.add_argument('--aggregate', default='plots/icu_aggregated.csv', help='Path to save aggregated CSV of results')
    args = parser.parse_args()
    results, agg_df = gather_results(args.dir, aggregate_out=args.aggregate)
    if not results:
        raise SystemExit('No matching results found in directory. Check filenames and location.')
    plot_results(results, args.out)
    if not agg_df.empty:
        print(f"Aggregated {len(agg_df)} result rows into {args.aggregate}")


if __name__ == '__main__':
    main()
