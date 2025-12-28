"""
Script to compare results from different Pearson delta thresholds
Usage: python compare_pearson_results.py
"""

import pandas as pd
import os
import glob

# Find all result files with pearson in the name
result_files = glob.glob("./exp_results/pearson_ICU_*_pearson_*.csv")

if not result_files:
    print("No result files found. Please run experiments first.")
    exit()

print(f"Found {len(result_files)} result files")
print()

# Extract delta values and results
results_summary = []

for file in sorted(result_files):
    # Extract delta from filename
    # Format: pearson_Airquality_10_SAGE_incre_alone_window_2_epoch_200_eval_0.1_pearson_0.X.csv
    filename = os.path.basename(file)
    parts = filename.split("_pearson_")
    if len(parts) == 2:
        delta_str = parts[1].replace(".csv", "")
        try:
            delta = float(delta_str)
        except:
            continue
        
        # Read the CSV file
        df = pd.read_csv(file, index_col=0)
        
        # Get the first row (window 0 results)
        if len(df) > 0:
            row = df.iloc[0]
            results_summary.append({
                'delta': delta,
                'opt_epoch': row['opt_epoch'],
                'MAE': row['opt_mae'],
                'MSE': row['mse'],
                'MAPE': row['mape'],
                'params_M': row['para'],
                'memory_KB': row['memo'],
                'time_min': row['opt_time'],
                'total_time_min': row['tot_time']
            })

# Create summary DataFrame
summary_df = pd.DataFrame(results_summary)
summary_df = summary_df.sort_values('delta')

print("="*80)
print("PEARSON DELTA COMPARISON RESULTS")
print("="*80)
print()
print(summary_df.to_string(index=False))
print()

# Find best results
best_mae_idx = summary_df['MAE'].idxmin()
best_mse_idx = summary_df['MSE'].idxmin()
best_mape_idx = summary_df['MAPE'].idxmin()

print("="*80)
print("BEST RESULTS")
print("="*80)
print(f"Best MAE:  delta={summary_df.loc[best_mae_idx, 'delta']:.1f}, MAE={summary_df.loc[best_mae_idx, 'MAE']:.4f}")
print(f"Best MSE:  delta={summary_df.loc[best_mse_idx, 'delta']:.1f}, MSE={summary_df.loc[best_mse_idx, 'MSE']:.4f}")
print(f"Best MAPE: delta={summary_df.loc[best_mape_idx, 'delta']:.1f}, MAPE={summary_df.loc[best_mape_idx, 'MAPE']:.4f}")
print()

# Save comparison to CSV
output_file = "./exp_results/pearson_delta_comparison.csv"
summary_df.to_csv(output_file, index=False)
print(f"Comparison saved to: {output_file}")
print()

# Plot if matplotlib is available
try:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # MAE vs Delta
    axes[0, 0].plot(summary_df['delta'], summary_df['MAE'], 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Pearson Threshold (δ)', fontsize=12)
    axes[0, 0].set_ylabel('MAE', fontsize=12)
    axes[0, 0].set_title('MAE vs Pearson Threshold', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # MSE vs Delta
    axes[0, 1].plot(summary_df['delta'], summary_df['MSE'], 'ro-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Pearson Threshold (δ)', fontsize=12)
    axes[0, 1].set_ylabel('MSE', fontsize=12)
    axes[0, 1].set_title('MSE vs Pearson Threshold', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # MAPE vs Delta
    axes[1, 0].plot(summary_df['delta'], summary_df['MAPE'], 'go-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Pearson Threshold (δ)', fontsize=12)
    axes[1, 0].set_ylabel('MAPE', fontsize=12)
    axes[1, 0].set_title('MAPE vs Pearson Threshold', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Time vs Delta
    axes[1, 1].plot(summary_df['delta'], summary_df['total_time_min'], 'mo-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Pearson Threshold (δ)', fontsize=12)
    axes[1, 1].set_ylabel('Total Time (minutes)', fontsize=12)
    axes[1, 1].set_title('Execution Time vs Pearson Threshold', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = "./exp_results/pearson_delta_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plots saved to: {plot_file}")
    print("To view plots, open the PNG file or uncomment plt.show()")
    # plt.show()
    
except ImportError:
    print("matplotlib not available. Skipping plots.")
    print("Install with: pip install matplotlib")
