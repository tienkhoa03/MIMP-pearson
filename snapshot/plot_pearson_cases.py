"""
Plot Pearson threshold comparison for multiple datasets and eval values.
Usage: python plot_pearson_cases.py
"""

import glob
import os
import pandas as pd

# Easy-to-edit configuration
DATASETS = ["Airquality", "ICU"]
EVAL_VALUES = [0.1, 0.3, 0.5]

EXP_RESULTS_DIR = "./exp_results"
FILE_TEMPLATE = "pearson_{dataset}_10_SAGE++DAMC_incre_alone_window_2_epoch_200_eval_{eval_value}_pearson_*.csv"


def collect_case_results(dataset: str, eval_value: float) -> pd.DataFrame:
    """Collect and summarize all delta results for one dataset/eval case."""
    case_dir = os.path.join(EXP_RESULTS_DIR, f"{dataset}_eval_{eval_value}_pearson")
    pattern = os.path.join(case_dir, FILE_TEMPLATE.format(dataset=dataset, eval_value=eval_value))
    result_files = glob.glob(pattern)

    if not result_files:
        print(f"[{dataset} | eval={eval_value}] No result files found at pattern: {pattern}")
        return pd.DataFrame()

    results_summary = []

    for file_path in sorted(result_files):
        filename = os.path.basename(file_path)
        parts = filename.split("_pearson_")
        if len(parts) != 2:
            continue

        delta_str = parts[1].replace(".csv", "")
        try:
            delta = float(delta_str)
        except ValueError:
            continue

        df = pd.read_csv(file_path, index_col=0)
        if len(df) == 0:
            continue

        row = df.iloc[0]
        results_summary.append(
            {
                "delta": delta,
                "opt_epoch": row["opt_epoch"],
                "MAE": row["opt_mae"],
                "MSE": row["mse"],
                "MAPE": row["mape"],
                "params_M": row["para"],
                "memory_KB": row["memo"],
                "time_min": row["opt_time"],
                "total_time_min": row["tot_time"],
            }
        )

    if not results_summary:
        print(f"[{dataset} | eval={eval_value}] Found files but no valid rows.")
        return pd.DataFrame()

    summary_df = pd.DataFrame(results_summary).sort_values("delta")
    return summary_df


def print_case_report(dataset: str, eval_value: float, summary_df: pd.DataFrame) -> None:
    """Print text report for one case."""
    print("=" * 90)
    print(f"PEARSON DELTA COMPARISON | DATASET={dataset} | EVAL={eval_value}")
    print("=" * 90)
    print(summary_df.to_string(index=False))
    print()

    best_mae_idx = summary_df["MAE"].idxmin()
    best_mse_idx = summary_df["MSE"].idxmin()
    best_mape_idx = summary_df["MAPE"].idxmin()

    print(f"Best MAE:  delta={summary_df.loc[best_mae_idx, 'delta']:.1f}, MAE={summary_df.loc[best_mae_idx, 'MAE']:.4f}")
    print(f"Best MSE:  delta={summary_df.loc[best_mse_idx, 'delta']:.1f}, MSE={summary_df.loc[best_mse_idx, 'MSE']:.4f}")
    print(f"Best MAPE: delta={summary_df.loc[best_mape_idx, 'delta']:.1f}, MAPE={summary_df.loc[best_mape_idx, 'MAPE']:.4f}")
    print()


def save_case_plot(dataset: str, eval_value: float, summary_df: pd.DataFrame) -> None:
    """Save one 2x2 plot figure for one dataset/eval case."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Skipping plots.")
        print("Install with: pip install matplotlib")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # MAE vs Delta
    axes[0, 0].plot(summary_df["delta"], summary_df["MAE"], "bo-", linewidth=2, markersize=8)
    axes[0, 0].set_xlabel("Pearson Threshold (δ)", fontsize=12)
    axes[0, 0].set_ylabel("MAE", fontsize=12)
    axes[0, 0].set_title("MAE vs Pearson Threshold", fontsize=14, fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3)

    # MSE vs Delta
    axes[0, 1].plot(summary_df["delta"], summary_df["MSE"], "ro-", linewidth=2, markersize=8)
    axes[0, 1].set_xlabel("Pearson Threshold (δ)", fontsize=12)
    axes[0, 1].set_ylabel("MSE", fontsize=12)
    axes[0, 1].set_title("MSE vs Pearson Threshold", fontsize=14, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)

    # MAPE vs Delta
    axes[1, 0].plot(summary_df["delta"], summary_df["MAPE"], "go-", linewidth=2, markersize=8)
    axes[1, 0].set_xlabel("Pearson Threshold (δ)", fontsize=12)
    axes[1, 0].set_ylabel("MAPE", fontsize=12)
    axes[1, 0].set_title("MAPE vs Pearson Threshold", fontsize=14, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3)

    # Time vs Delta
    axes[1, 1].plot(summary_df["delta"], summary_df["total_time_min"], "mo-", linewidth=2, markersize=8)
    axes[1, 1].set_xlabel("Pearson Threshold (δ)", fontsize=12)
    axes[1, 1].set_ylabel("Total Time (minutes)", fontsize=12)
    axes[1, 1].set_title("Execution Time vs Pearson Threshold", fontsize=14, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"Pearson Delta Comparison | {dataset} | eval={eval_value}", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_png = os.path.join(EXP_RESULTS_DIR, f"pearson_delta_comparison_{dataset}_eval_{eval_value}.png")
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[{dataset} | eval={eval_value}] Plot saved to: {output_png}")


def main() -> None:
    any_success = False

    for dataset in DATASETS:
        for eval_value in EVAL_VALUES:
            summary_df = collect_case_results(dataset, eval_value)
            if summary_df.empty:
                continue

            any_success = True
            print_case_report(dataset, eval_value, summary_df)

            output_csv = os.path.join(EXP_RESULTS_DIR, f"pearson_delta_comparison_{dataset}_eval_{eval_value}.csv")
            summary_df.to_csv(output_csv, index=False)
            print(f"[{dataset} | eval={eval_value}] Summary saved to: {output_csv}")

            save_case_plot(dataset, eval_value, summary_df)
            print()

    if not any_success:
        print("No valid results found for any configured dataset/eval case.")


if __name__ == "__main__":
    main()
