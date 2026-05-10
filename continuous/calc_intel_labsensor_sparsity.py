"""Calculate sparsity of the Intel Berkeley Research Lab sensor dataset.

The script uses the existing dataset loader in `utils/load_dataset.py`, so the
definition of sparsity matches the current project pipeline:

    sparsity = number_of_missing_values / total_number_of_values

It also reports per-feature sparsity for temperature, humidity, and light.

Usage:
    python calc_intel_labsensor_sparsity.py

Optional:
    python calc_intel_labsensor_sparsity.py --out plots/intel_labsensor_sparsity.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_FILE = os.path.join(REPO_ROOT, "data", "intel_berkeley_research_lab_sensor_data.txt")


FEATURE_NAMES = ["temperature", "humidity", "light"]


def load_intel_labsensor_matrix() -> np.ndarray:
    """Load the Intel Berkeley Research Lab sensor data into a dense matrix.

    This is a local copy of the project loader logic, but without the optional
    `pypots` dependency so the sparsity calculator can run standalone.
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
    df = df[(df["temperature"] > 0) & (df["temperature"] < 50)]
    df = df[(df["humidity"] > 0) & (df["humidity"] < 100)]
    df = df[(df["voltage"] > 2.0) & (df["voltage"] < 3.0)]
    df = df.sort_values(["epoch", "moteid"]).reset_index(drop=True)

    epochs = np.sort(df["epoch"].unique())
    moteids = np.sort(df["moteid"].unique())
    matrix = np.full((len(epochs), len(moteids), len(FEATURE_NAMES)), np.nan, dtype=float)

    for idx, feature_name in enumerate(FEATURE_NAMES):
        pivot = (
            df.pivot_table(index="epoch", columns="moteid", values=feature_name, aggfunc="mean")
            .reindex(index=epochs, columns=moteids)
            .to_numpy()
        )
        matrix[:, :, idx] = pivot

    return matrix.reshape(-1, len(FEATURE_NAMES))


def compute_sparsity(matrix: np.ndarray) -> dict[str, float]:
    total_values = matrix.size
    missing_values = int(np.isnan(matrix).sum())
    observed_values = total_values - missing_values

    summary: dict[str, float] = {
        "num_rows": int(matrix.shape[0]),
        "num_cols": int(matrix.shape[1]),
        "total_values": int(total_values),
        "missing_values": missing_values,
        "observed_values": int(observed_values),
        "missing_ratio": missing_values / total_values if total_values else 0.0,
        "observed_ratio": observed_values / total_values if total_values else 0.0,
    }

    for idx, feature_name in enumerate(FEATURE_NAMES):
        feature_col = matrix[:, idx]
        feature_total = feature_col.size
        feature_missing = int(np.isnan(feature_col).sum())
        summary[f"{feature_name}_missing_values"] = feature_missing
        summary[f"{feature_name}_missing_ratio"] = (
            feature_missing / feature_total if feature_total else 0.0
        )

    return summary


def save_summary(summary: dict[str, float], out_file: str) -> None:
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    pd.DataFrame([summary]).to_csv(out_file, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate Intel labsensor sparsity.")
    parser.add_argument(
        "--out",
        default=os.path.join("plots", "intel_labsensor_sparsity.csv"),
        help="Output CSV path under continuous/",
    )
    args = parser.parse_args()

    matrix = load_intel_labsensor_matrix()

    summary = compute_sparsity(matrix)
    save_summary(summary, os.path.join(SCRIPT_DIR, args.out))

    print("Intel Berkeley Research Lab sensor sparsity summary")
    print(f"Rows: {summary['num_rows']}")
    print(f"Columns: {summary['num_cols']}")
    print(f"Missing values: {summary['missing_values']}")
    print(f"Observed values: {summary['observed_values']}")
    print(f"Missing ratio: {summary['missing_ratio']:.6f}")
    print(f"Observed ratio: {summary['observed_ratio']:.6f}")
    for feature_name in FEATURE_NAMES:
        ratio = summary[f"{feature_name}_missing_ratio"]
        missing = summary[f"{feature_name}_missing_values"]
        print(f"{feature_name}: missing={missing}, ratio={ratio:.6f}")
    print(f"Saved CSV to {os.path.join(SCRIPT_DIR, args.out)}")


if __name__ == "__main__":
    main()
