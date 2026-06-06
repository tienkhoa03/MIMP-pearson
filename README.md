# MIMPpearson

**Missing data Imputation via Message Passing with Pearson-filtered graphs**

A GNN-based missing data imputation framework that builds dynamic similarity graphs from sensor/time-series data. Extends the original MPIN architecture with GraphSAGE++ variants and optional Pearson correlation filtering for graph construction.

---

## Overview

Missing data is a common challenge in real-world sensor deployments (ICU monitors, air quality stations, WiFi fingerprinting, building sensors). This project proposes and evaluates:

- **GraphSAGE++ variants** (DA, DAC, DAMC) — progressively richer message-passing aggregations
- **Pearson-filtered graph construction** — connects nodes via KNN in the subspace of correlated features, rather than across all features blindly
- **Two imputation modes** — snapshot (single-window) and continuous (streaming/incremental)

Baselines include Feature Propagation (FP), MICE, KNN imputation, SVD, SAITS, and BRITS.

---

## Project Structure

```
MIMPpearson/
├── snapshot/                   # Single-window imputation experiments
│   ├── MPIN-plus.py            # Main entry point (Pearson + GraphSAGE++ support)
│   ├── MPIN.py                 # Original MPIN implementation
│   ├── FP.py                   # Feature Propagation baseline
│   ├── trad.py                 # Traditional baselines (MICE, KNN, SVD)
│   ├── nn.py                   # Neural baselines (SAITS, BRITS)
│   ├── run_*_experiments.ps1   # PowerShell experiment scripts (Windows)
│   ├── run_*_experiments.sh    # Bash experiment scripts (Linux/macOS)
│   ├── exp_results/            # Output CSVs organised by dataset
│   └── plots/                  # Aggregated results and comparison plots
│
├── continuous/                 # Streaming/incremental imputation
│   ├── continuous.py           # Main entry point for continuous mode
│   ├── run_*_experiments.ps1   # Experiment automation scripts
│   ├── iif_exp/                # Incremental imputation feature experiments
│   └── plots/                  # Aggregated streaming results
│
├── utils/
│   ├── DynamicGNN.py           # GNN architectures (GAT, GCN, SAGE, SAGE++ variants)
│   ├── similarity_graph.py     # Pearson-filtered graph construction
│   ├── regressor.py            # MLP regressor head
│   ├── load_dataset.py         # Dataset loaders (ICU, Airquality, WiFi, Labsensor, Weather)
│   └── load_dataset_synth.py   # Synthetic dataset generation
│
├── sageplus/                   # GraphSAGE++ model definitions
│   ├── GraphSAGE++DA.py        # Dual Aggregation
│   ├── GraphSAGE++DAC.py       # Dual Aggregation + Concatenation
│   └── GraphSAGE++DAMC.py      # Dual Aggregation + Mean-Max Concatenation
│
├── data/                       # Raw datasets (not committed, see below)
├── app/                        # Downstream classification on imputed data
├── requirements.txt
└── note.txt                    # Developer notes and example commands
```

---

## Datasets

| Name | Key | Description | Features |
|------|-----|-------------|----------|
| PhysioNet ICU 2012 | `ICU` | ICU patient vital signs (48h records) | 37 clinical features |
| Beijing Air Quality | `Airquality` | Multi-station air pollution readings | 11 pollutant/weather features |
| WiFi Fingerprint | `KDM` / `WDS` / `LHS` | RSS measurements for indoor localisation | ~200 APs |
| Intel Labsensor (IBRL) | `Labsensor` | Building sensor readings | temperature, humidity, light |
| Weather | `Weather` | Meteorological time series | multiple weather variables |

The Labsensor raw file (`intel_berkeley_research_lab_sensor_data.txt`) must be placed in `data/` manually — it is excluded from git due to its size (~150 MB).

ICU and Airquality datasets are downloaded automatically on first run via PyPOTS/TSDB.

---

## Installation

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Linux/macOS
.\venv\Scripts\Activate.ps1       # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Install PyTorch and PyTorch Geometric separately
# (choose the CUDA version that matches your system)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.x.x+cu128.html
```

---

## Quick Start

Run snapshot imputation on the Labsensor dataset with GraphSAGE++DAMC:

```bash
cd snapshot

# Standard KNN graph (no Pearson)
python MPIN-plus.py \
    --dataset Labsensor \
    --base SAGE++DAMC \
    --epochs 200 \
    --k 10 \
    --eval_ratio 0.3 \
    --prefix no_pearson

# Feature-Pearson filtered graph
python MPIN-plus.py \
    --dataset Labsensor \
    --base SAGE++DAMC \
    --epochs 200 \
    --k 10 \
    --eval_ratio 0.3 \
    --use_pearson true \
    --delta 0.5 \
    --prefix pearson

# Pearson with fallback disabled (empty graph when no correlated features found)
python MPIN-plus.py \
    --dataset Labsensor \
    --base SAGE++DAMC \
    --use_pearson true \
    --delta 0.9 \
    --fallback_knn false \
    --prefix pearson_strict
```

---

## Arguments — `snapshot/MPIN-plus.py`

### Data

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `ICU` | Dataset name: `ICU`, `Airquality`, `KDM`, `WDS`, `LHS`, `Labsensor`, `Weather` |
| `--window` | `2` | Time window length in minutes (WiFi) or number of timesteps |
| `--stream` | `1.0` | Fraction of dataset to use (e.g. `0.5` = first 50%) |
| `--eval_ratio` | `0.1` | Fraction of observed values held out for evaluation |

### Model

| Argument | Default | Description |
|----------|---------|-------------|
| `--base` | `SAGE` | GNN backbone: `GCN`, `GAT`, `SAGE`, `SAGE++DA`, `SAGE++DAC`, `SAGE++DAMC` |
| `--dynamic` | `false` | Rebuild graph at each forward pass (`true`) or use fixed graph (`false`) |
| `--out_channels` | `256` | Hidden dimension of GNN layers |
| `--k` | `10` | Number of KNN neighbours for graph construction |

### Training

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | `200` | Number of training epochs per window |
| `--lr` | `0.01` | Learning rate |
| `--weight_decay` | `0.1` | Adam weight decay |
| `--num_of_iter` | `5` | Number of independent runs (results are averaged) |

### Graph Construction

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_pearson` | `false` | Enable feature-Pearson filtered graph construction |
| `--delta` | `0.3` | Pearson correlation threshold — only features with `\|r\| >= delta` are used for KNN |
| `--fallback_knn` | `true` | Fall back to standard KNN when no feature pairs pass the delta threshold |

### Output

| Argument | Default | Description |
|----------|---------|-------------|
| `--prefix` | `testKnnK` | Prefix for output CSV filename |
| `--incre_mode` | `alone` | Incremental learning mode (currently only `alone` is active) |

---

## GNN Backbones

| Backbone | Description |
|----------|-------------|
| `GCN` | Graph Convolutional Network |
| `GAT` | Graph Attention Network |
| `SAGE` | GraphSAGE with mean aggregation |
| `SAGE++DA` | Dual Aggregation — combines mean and max |
| `SAGE++DAC` | DA + concatenates the two aggregations before projection |
| `SAGE++DAMC` | DAC + uses separate MLPs for mean and max paths before concatenation |

---

## Pearson-Filtered Graph Construction

The `--use_pearson true` mode builds the graph in two steps:

1. **Feature correlation matrix** — compute Pearson correlation between each pair of feature time series (columns of X). For Labsensor this yields a 3×3 matrix for temperature, humidity, and light.

2. **Correlated feature subspace KNN** — retain only features involved in at least one pair with `|r| >= delta`, then run KNN using only those features as the embedding space.

This ensures edges connect nodes that are similar in features that actually co-vary, rather than all features equally.

When `--fallback_knn true` (default), the graph falls back to standard all-feature KNN if no feature pair passes the threshold. Set `--fallback_knn false` to keep an empty graph instead — useful for ablation studies where you want to isolate the effect of having no graph signal.

---

## Output Format

Results are written to `snapshot/exp_results/` as CSV files with the naming convention:

```
{prefix}_{dataset}_{k}_{base}_incre_{mode}_window_{window}_epoch_{epochs}_eval_{eval_ratio}[_pearson_{delta}].csv
```

Each row corresponds to one time window. Columns:

| Column | Description |
|--------|-------------|
| `opt_epoch` | Epoch with the best validation MAE |
| `opt_mae` | Best validation MAE |
| `mse` | MSE at the best MAE epoch |
| `mape` | MRE (Mean Relative Error) at the best MAE epoch |
| `para` | Number of trainable parameters (millions) |
| `memo` | Total model size (MB) |
| `opt_time` | Elapsed time at the best epoch (minutes) |
| `tot_time` | Total training time (minutes) |

---

## Ablation Studies

Key dimensions to vary for ablation:

```bash
# 1. Graph construction: KNN vs Pearson-filtered
--use_pearson false
--use_pearson true --delta 0.3

# 2. Pearson threshold sensitivity
--use_pearson true --delta 0.1
--use_pearson true --delta 0.3
--use_pearson true --delta 0.5
--use_pearson true --delta 0.7
--use_pearson true --delta 0.9

# 3. GNN backbone comparison
--base SAGE
--base SAGE++DA
--base SAGE++DAC
--base SAGE++DAMC
--base GAT
--base GCN

# 4. Dynamic vs static graph
--dynamic true
--dynamic false

# 5. Number of KNN neighbours
--k 5
--k 10
--k 15
--k 20

# 6. Fallback behaviour (Pearson mode only)
--use_pearson true --fallback_knn true    # degrade gracefully to KNN
--use_pearson true --fallback_knn false   # strict — no fallback
```

Batch experiment scripts for each dataset are available in `snapshot/run_*_experiments.ps1` (Windows) and `snapshot/run_*_experiments.sh` (Linux/macOS).

---

## Continuous Mode

The `continuous/continuous.py` script supports streaming imputation where the model is updated incrementally as new data arrives. It shares the same arguments as the snapshot mode and outputs results to `continuous/iif_exp/`.

```bash
cd continuous
python continuous.py \
    --dataset Labsensor \
    --base SAGE++DAMC \
    --epochs 100 \
    --stream 0.5
```

---

## Evaluated Metrics

- **MAE** — Mean Absolute Error on artificially-masked held-out values
- **MSE** — Mean Squared Error on held-out values
- **MRE** — Mean Relative Error (reported as MAPE in some scripts)
- **Model size** — Total parameter count and memory footprint
- **Training time** — Wall-clock time to reach the best epoch
