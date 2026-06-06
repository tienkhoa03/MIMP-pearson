"""
Thống kê tỉ lệ fallback KNN khi Pearson filtering không cho ra cặp stream nào.

Config: dataset=ICU (PhysioNet 2012), load_window=48, stream=0.1, pearson_window=4
        delta: 0.9 -> 0.1 (step 0.1)
        eval_ratio: 0.1, 0.3, 0.5
        num_of_iter=5

Đầu ra: exp_results/pearson_fallback_stats_icu.csv
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import csv
import pickle
import random

import numpy as np
import torch

from utils.similarity_graph import compute_stream_pearson

# ── Config ──────────────────────────────────────────────────────────────────
STREAM         = 0.1   # fraction of ICU patients to use
LOAD_WINDOW    = 48    # timesteps per patient (max 48 in PhysioNet 2012)
PEARSON_WINDOW = 4     # group size for stream Pearson (4 timesteps = 4 hours)
NUM_ITER       = 5
DELTAS         = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
EVAL_RATIOS    = [0.1, 0.3, 0.5]
OUT_CSV        = os.path.join(
    os.path.dirname(__file__),
    "exp_results",
    "pearson_fallback_stats_icu.csv",
)
# ────────────────────────────────────────────────────────────────────────────


def load_icu_direct(stream=0.1, load_window=48):
    """Load physionet_2012 từ tsdb pkl cache, bỏ qua pypots (pandas 2.0 incompatible)."""
    cache_path = os.path.join(
        os.path.expanduser("~"), ".tsdb_cached_datasets",
        "physionet_2012", "physionet_2012_cache.pkl"
    )
    with open(cache_path, "rb") as f:
        data = pickle.load(f)

    X_df = data["X"].copy()
    static_features = set(data.get("static_features", []))

    # Feature columns: bỏ RecordID, Time và static features
    meta_cols = {"RecordID", "Time"} | static_features
    feature_cols = [c for c in X_df.columns if c not in meta_cols]

    # Shuffle và lấy một phần bệnh nhân
    record_ids = X_df["RecordID"].unique()
    rng = np.random.default_rng(seed=42)
    rng.shuffle(record_ids)
    n_use = max(1, int(len(record_ids) * stream))
    selected_ids = record_ids[:n_use]

    X_sub = X_df[X_df["RecordID"].isin(set(selected_ids))].copy()
    X_sub = X_sub.sort_values(["RecordID", "Time"])

    # Pad/truncate từng bệnh nhân thành đúng load_window dòng (vectorized)
    n_feat = len(feature_cols)
    out = np.full((n_use * load_window, n_feat), np.nan)
    for i, rid in enumerate(selected_ids):
        rows = X_sub.loc[X_sub["RecordID"] == rid, feature_cols].to_numpy(dtype=float)
        n = min(len(rows), load_window)
        out[i * load_window : i * load_window + n] = rows[:n]

    return out   # (n_use * load_window, n_features)


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
print("Đang nạp ICU (PhysioNet 2012) từ tsdb cache...")
base_X = load_icu_direct(stream=STREAM, load_window=LOAD_WINDOW)
base_X_mask = (~np.isnan(base_X)).astype(int)
base_X = base_X.copy()

feature_means = np.nanmean(base_X, axis=0)
for col in range(base_X.shape[1]):
    base_X[np.isnan(base_X[:, col]), col] = feature_means[col]
mean_X = np.mean(base_X)
std_X = np.std(base_X)
base_X = (base_X - mean_X) / std_X

num_streams_est = base_X.shape[0] // PEARSON_WINDOW
print(
    f"Shape sau tien xu ly: {base_X.shape}  "
    f"(features={base_X.shape[1]}, pearson_window={PEARSON_WINDOW}, "
    f"~{num_streams_est} streams)"
)

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
            if check_fallback(X_filled, delta, PEARSON_WINDOW):
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

print(f"\nKet qua da ghi vao: {OUT_CSV}")
