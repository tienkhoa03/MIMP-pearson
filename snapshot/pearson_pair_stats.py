"""
Thống kê phân phối tương quan Pearson giữa các stream qua nhiều lần chạy.

Với mỗi eval_ratio, chạy N_RUNS lần (mỗi lần data_transform ngẫu nhiên khác nhau),
thu thập tất cả giá trị Pearson ngoài đường chéo của ma trận stream × stream,
rồi tính: mean, max, min, variance, std trên mỗi lần chạy → tổng hợp qua N_RUNS.

Config: dataset=Labsensor, window=30, stream=0.01
        eval_ratio: 0.1, 0.3, 0.5
        N_RUNS=100

Đầu ra: exp_results/pearson_pair_stats_labsensor.csv
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import csv
import random

import numpy as np
import torch

from utils.load_dataset import load_IBRL_dataset
from utils.similarity_graph import compute_stream_pearson

# ── Config ──────────────────────────────────────────────────────────────────
STREAM      = 0.1
WINDOW      = 30
N_RUNS      = 100
EVAL_RATIOS = [0.1, 0.3, 0.5]
OUT_CSV     = os.path.join(
    os.path.dirname(__file__),
    "exp_results",
    "pearson_pair_stats_labsensor.csv",
)
# ────────────────────────────────────────────────────────────────────────────


def data_transform(X, X_mask, eval_ratio):
    X = X.copy()
    X_mask = X_mask.copy()
    rows, cols = np.where(X_mask == 1)
    n_eval = int(eval_ratio * len(rows))
    if n_eval > 0:
        idx = random.sample(range(len(rows)), n_eval)
        er, ec = rows[idx], cols[idx]
        X_mask[er, ec] = 0
        X[er, ec] = 0
    return X, X_mask


def fill_missing(X, X_mask):
    X = X.copy()
    for col in range(X.shape[1]):
        obs = X_mask[:, col] == 1
        if obs.any():
            X[~obs, col] = X[obs, col].mean()
    return X


def offdiag_stats(X_np, window_length):
    """Trả về (mean, max, min, var, std) của các giá trị ngoài đường chéo ma trận Pearson stream."""
    p = torch.FloatTensor(X_np)
    pearson, num_streams = compute_stream_pearson(p, window_length=window_length)
    mask = ~torch.eye(num_streams, dtype=torch.bool)
    vals = pearson[mask].numpy()
    return float(np.mean(vals)), float(np.max(vals)), float(np.min(vals)), float(np.var(vals)), float(np.std(vals))


# ── Nạp và tiền xử lý dữ liệu ───────────────────────────────────────────────
print("Đang nạp Labsensor...")
base_X = load_IBRL_dataset(method="mpin")
base_X_mask = (~np.isnan(base_X)).astype(int)
base_X = base_X.copy()

num_rows = max(1, int(base_X.shape[0] * STREAM))
print(f"Stream={STREAM}: giữ {num_rows}/{base_X.shape[0]} dòng đầu")
base_X = base_X[:num_rows]
base_X_mask = base_X_mask[:num_rows]

feature_means = np.nanmean(base_X, axis=0)
for col in range(base_X.shape[1]):
    base_X[np.isnan(base_X[:, col]), col] = feature_means[col]
mean_X = np.mean(base_X)
std_X  = np.std(base_X)
base_X = (base_X - mean_X) / std_X

# Tính số streams thực tế
num_streams_actual = num_rows // WINDOW
print(f"Shape sau tiền xử lý: {base_X.shape}  window={WINDOW}  num_streams={num_streams_actual}")

# ── Vòng thực nghiệm ─────────────────────────────────────────────────────────
random.seed(2021)
results = []

for eval_ratio in EVAL_RATIOS:
    # Mỗi run cho ra (mean, max, min, var, std) của ma trận Pearson stream
    run_means = []
    run_maxs  = []
    run_mins  = []
    run_vars  = []
    run_stds  = []

    for run in range(N_RUNS):
        X, X_mask = data_transform(base_X, base_X_mask, eval_ratio)
        X_filled  = fill_missing(X, X_mask)
        m, mx, mn, v, s = offdiag_stats(X_filled, WINDOW)
        run_means.append(m)
        run_maxs.append(mx)
        run_mins.append(mn)
        run_vars.append(v)
        run_stds.append(s)

    run_means = np.array(run_means)
    run_maxs  = np.array(run_maxs)
    run_mins  = np.array(run_mins)

    sep = "=" * 65
    print(f"\n{sep}")
    print(f"eval_ratio={eval_ratio}  ({N_RUNS} runs)  num_streams={num_streams_actual}")
    print(f"{sep}")
    print(f"  Phân phối Pearson stream (ngoài đường chéo):")
    print(f"  {'Chỉ số':<20}  {'mean':>10}  {'std':>10}  {'min':>10}  {'max':>10}")
    print(f"  {'-'*18}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    print(f"  {'mean_per_run':<20}  {run_means.mean():+10.4f}  {run_means.std():10.6f}  {run_means.min():+10.4f}  {run_means.max():+10.4f}")
    print(f"  {'max_per_run':<20}  {run_maxs.mean():+10.4f}  {run_maxs.std():10.6f}  {run_maxs.min():+10.4f}  {run_maxs.max():+10.4f}")
    print(f"  {'min_per_run':<20}  {run_mins.mean():+10.4f}  {run_mins.std():10.6f}  {run_mins.min():+10.4f}  {run_mins.max():+10.4f}")

    results.append({
        "eval_ratio":        eval_ratio,
        "n_runs":            N_RUNS,
        "num_streams":       num_streams_actual,
        "mean_of_means":     round(float(run_means.mean()), 6),
        "std_of_means":      round(float(run_means.std()),  6),
        "mean_of_maxs":      round(float(run_maxs.mean()),  6),
        "std_of_maxs":       round(float(run_maxs.std()),   6),
        "mean_of_mins":      round(float(run_mins.mean()),  6),
        "std_of_mins":       round(float(run_mins.std()),   6),
    })

# ── Ghi CSV ──────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
fields = [
    "eval_ratio", "n_runs", "num_streams",
    "mean_of_means", "std_of_means",
    "mean_of_maxs",  "std_of_maxs",
    "mean_of_mins",  "std_of_mins",
]
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)

print(f"\nKết quả đã ghi vào: {OUT_CSV}")
