"""
Thống kê tỉ lệ fallback KNN khi Pearson filtering không cho ra cặp stream nào.

Config: dataset=Labsensor, window=30, stream=0.01
        delta: 0.9 -> 0.1 (step 0.1)
        eval_ratio: 0.1, 0.3, 0.5
        num_of_iter=5

Đầu ra: exp_results/pearson_fallback_stats_labsensor.csv
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import copy
import csv
import random

import numpy as np
import torch

from utils.load_dataset import load_IBRL_dataset
from utils.similarity_graph import compute_stream_pearson

# ── Config ──────────────────────────────────────────────────────────────────
STREAM      = 0.1
WINDOW      = 30
NUM_ITER    = 5
DELTAS      = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
EVAL_RATIOS = [0.1, 0.3, 0.5]
OUT_CSV     = os.path.join(
    os.path.dirname(__file__),
    "exp_results",
    "pearson_fallback_stats_labsensor.csv",
)
# ────────────────────────────────────────────────────────────────────────────


def data_transform(X, X_mask, eval_ratio):
    X = X.copy()
    X_mask = X_mask.copy()
    eval_mask = np.zeros(X_mask.shape)
    rows, cols = np.where(X_mask == 1)
    n_eval = int(eval_ratio * len(rows))
    if n_eval == 0:
        return X, X_mask, eval_mask
    idx = random.sample(range(len(rows)), n_eval)
    er, ec = rows[idx], cols[idx]
    X_mask[er, ec] = 0
    eval_mask[er, ec] = 1
    X[er, ec] = 0
    return X, X_mask, eval_mask


def fill_missing(X, X_mask):
    X = X.copy()
    for col in range(X.shape[1]):
        obs = X_mask[:, col] == 1
        if obs.any():
            X[~obs, col] = X[obs, col].mean()
    return X


def check_fallback(X_np, delta, window_length):
    """True khi không có cặp stream nào có |pearson| >= delta."""
    p = torch.FloatTensor(X_np)
    pearson, _ = compute_stream_pearson(p, window_length=window_length)
    mask = pearson.abs() >= delta
    mask.fill_diagonal_(False)
    return not mask.any().item()


# ── Nạp và tiền xử lý dữ liệu ───────────────────────────────────────────────
print("Đang nạp Labsensor...")
base_X = load_IBRL_dataset(method="mpin")
base_X_mask = (~np.isnan(base_X)).astype(int)
base_X = base_X.copy()

num_rows = max(1, int(base_X.shape[0] * STREAM))
print(f"Stream={STREAM}: giữ {num_rows}/{base_X.shape[0]} dòng đầu")
base_X = base_X[:num_rows]
base_X_mask = base_X_mask[:num_rows]

# Điền NaN bằng mean cột, rồi chuẩn hoá toàn cục
feature_means = np.nanmean(base_X, axis=0)
for col in range(base_X.shape[1]):
    base_X[np.isnan(base_X[:, col]), col] = feature_means[col]
mean_X = np.mean(base_X)
std_X = np.std(base_X)
base_X = (base_X - mean_X) / std_X

print(f"Shape sau tiền xử lý: {base_X.shape}  (features={base_X.shape[1]}, window={WINDOW})")

# ── Vòng thực nghiệm ──────────────────────────────────────────────────────────
random.seed(2021)
results = []
total = len(DELTAS) * len(EVAL_RATIOS)
current = 0

for delta in DELTAS:
    for eval_ratio in EVAL_RATIOS:
        current += 1
        fallback_count = 0

        for _ in range(NUM_ITER):
            X, X_mask, _ = data_transform(base_X, base_X_mask, eval_ratio)
            X_filled = fill_missing(X, X_mask)
            if check_fallback(X_filled, delta, WINDOW):
                fallback_count += 1

        fallback_pct = round(100.0 * fallback_count / NUM_ITER, 2)
        print(
            f"[{current:2d}/{total}] delta={delta:.1f}  eval_ratio={eval_ratio}"
            f"  fallback={fallback_count}/{NUM_ITER} ({fallback_pct:.0f}%)"
        )
        results.append(
            {
                "delta": delta,
                "eval_ratio": eval_ratio,
                "fallback_count": fallback_count,
                "num_iter": NUM_ITER,
                "fallback_pct": fallback_pct,
            }
        )

# ── Ghi CSV ──────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
fields = ["delta", "eval_ratio", "fallback_count", "num_iter", "fallback_pct"]
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)

print(f"\nKết quả đã ghi vào: {OUT_CSV}")
