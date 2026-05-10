"""Aggregate CSV result files under continuous/iif_exp/Labsensor into one CSV.

Usage:
  python plot_labsensor.py [--src DIR] [--out FILE]

Defaults:
  src: continuous/iif_exp/Labsensor
  out: continuous/plots/labsensor_aggregated.csv
"""
import argparse
import glob
import os

import pandas as pd


def find_csv_files(src_dir):
    pattern = os.path.join(src_dir, "**", "*.csv")
    return glob.glob(pattern, recursive=True)


def aggregate(src_dir, out_file):
    files = find_csv_files(src_dir)
    if not files:
        print(f"No CSV files found in {src_dir}")
        return

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"Skipping {f}: read error: {e}")
            continue
        rel = os.path.relpath(f)
        df.insert(0, "source_file", rel)
        dfs.append(df)

    if not dfs:
        print("No CSVs could be read successfully.")
        return

    out = pd.concat(dfs, ignore_index=True)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    out.to_csv(out_file, index=False)
    print(f"Wrote aggregated CSV to {out_file} ({len(out)} rows from {len(dfs)} files)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=os.path.join("iif_exp", "Labsensor"))
    p.add_argument("--out", default=os.path.join("plots", "labsensor_aggregated.csv"))
    args = p.parse_args()
    aggregate(args.src, args.out)


if __name__ == "__main__":
    main()