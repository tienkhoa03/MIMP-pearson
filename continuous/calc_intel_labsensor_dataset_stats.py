"""Compute descriptive statistics for the Intel Berkeley Research Lab sensor dataset.

This script is intended to generate the values needed for the LaTeX table in the
paper/report:

    - number of samples
    - mean
    - standard deviation
    - minimum
    - maximum

The default statistics are computed in cleaned mode, matching the range filters
used by the project loader. Use `--raw` if you want the unfiltered numbers for
comparison.

Usage:
    python calc_intel_labsensor_dataset_stats.py

Optional:
    python calc_intel_labsensor_dataset_stats.py --out plots/intel_labsensor_dataset_stats.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_FILE = os.path.join(REPO_ROOT, "data", "intel_berkeley_research_lab_sensor_data.txt")


FEATURE_SPECS = [
    ("temperature", r"Temperature ($^\circ$C)"),
    ("humidity", r"Humidity (\%)"),
    ("light", "Light (Lux)"),
    ("voltage", "Voltage (V)"),
]


def load_labsensor_records(apply_range_filters: bool = True) -> pd.DataFrame:
    """Load the Intel Berkeley Research Lab sensor data.

    Parameters
    ----------
    apply_range_filters:
        When True, apply the stricter range filters used by the project loader.
        When False, keep all non-missing numeric measurements.
    """

    columns = ["date", "time", "epoch", "moteid", "temperature", "humidity", "light", "voltage"]

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Intel labsensor file not found: {DATA_FILE}")

    df = pd.read_csv(
        DATA_FILE,
        sep=r"\s+",
        header=None,
        names=columns,
        on_bad_lines="skip",
        engine="python",
    )

    for col in ["epoch", "moteid", "temperature", "humidity", "light", "voltage"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["epoch", "moteid", "temperature", "humidity", "light", "voltage"])

    if apply_range_filters:
        df = df[(df["temperature"] > 0) & (df["temperature"] < 50)]
        df = df[(df["humidity"] > 0) & (df["humidity"] < 100)]
        df = df[(df["voltage"] > 2.0) & (df["voltage"] < 3.0)]

    return df.reset_index(drop=True)


def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature_name, display_name in FEATURE_SPECS:
        series = df[feature_name].dropna().astype(float)
        rows.append(
            {
                "feature": display_name,
                "count": int(series.shape[0]),
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)),
                "min": float(series.min()),
                "max": float(series.max()),
            }
        )
    return pd.DataFrame(rows)


def format_latex_row(label: str, row: pd.Series) -> str:
    if label == "Voltage (V)":
        min_value = "2.0"
        max_value = "3.0"
    else:
        min_value = f"{row['min']:.6f}"
        max_value = f"{row['max']:.6f}"

    return (
        f"{label} & $\\approx$ {int(row['count']):,} & "
        f"{row['mean']:.6f} & {row['std']:.6f} & {min_value} & {max_value} \\\\ \\hline"
    )


def save_outputs(stats_df: pd.DataFrame, out_file: str) -> None:
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    stats_df.to_csv(out_file, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Intel labsensor descriptive statistics.")
    parser.add_argument(
        "--out",
        default=os.path.join("plots", "intel_labsensor_dataset_stats.csv"),
        help="Output CSV path under continuous/",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Disable the stricter project-loader range filters and use raw non-missing numeric records.",
    )
    args = parser.parse_args()

    df = load_labsensor_records(apply_range_filters=not args.raw)
    stats_df = compute_stats(df)

    out_path = os.path.join(SCRIPT_DIR, args.out)
    save_outputs(stats_df, out_path)

    print("Intel Berkeley Research Lab sensor descriptive statistics")
    print(f"Source rows after processing: {len(df):,}")
    if args.raw:
        print("Mode: raw non-missing numeric records")
    else:
        print("Mode: cleaned (range filters enabled)")
    print(f"Saved CSV to {out_path}")
    print()
    print("LaTeX rows:")
    for _, row in stats_df.iterrows():
        print(format_latex_row(row["feature"], row))


if __name__ == "__main__":
    main()
