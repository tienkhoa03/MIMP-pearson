# Labsensor (Intel Berkeley Research Lab) Continuous Experiments

## Overview
This document explains how to run continuous imputation experiments using the Labsensor dataset (Intel Berkeley Research Lab sensor data).

## Dataset Information
- **Dataset Name**: Intel Berkeley Research Lab (IBRL) Sensor Data  
- **Features**: 3 (temperature, humidity, light)
- **Sensors (motes)**: 54 wireless sensor nodes
- **Time Points**: Multiple epochs with varying missingness patterns
- **Data Location**: `data/intel_berkeley_research_lab_sensor_data.txt`

## Running Experiments

### Bash (Linux/Mac)
```bash
cd continuous
bash run_labsensor_experiments.sh
```

### PowerShell (Windows)
```powershell
cd continuous
.\run_labsensor_experiments.ps1
```

### Manual Execution
To run a single experiment manually:
```bash
python continuous.py \
    --dataset Labsensor \
    --window 24 \
    --base SAGE++DAMC \
    --eval_ratio 0.5 \
    --epochs 200 \
    --use_pearson false
```

## Configuration Parameters

### Key Parameters for Labsensor
- `--dataset Labsensor`: Select the labsensor dataset
- `--window 24`: Window size in epochs (default for labsensor)
- `--eval_ratio`: Evaluation ratio (0.1, 0.3, 0.5)
- `--base`: GNN base model (e.g., SAGE++DAMC)
- `--epochs`: Number of training epochs (default: 200)
- `--use_pearson`: Enable Pearson-filtered graph construction (true/false)
- `--delta`: Pearson correlation threshold (0.1-0.9)
- `--stream`: Stream ratio for data sampling (default: 1.0)
- `--k`: Number of nearest neighbors (default: 10)
- `--num_of_iter`: Number of iterations (default: 1)

## Experiment Results
Results are saved in the `iif_exp/` directory with filenames following the pattern:
- No Pearson: `no_pearson_Labsensor_10_SAGE++DAMC_..._eval_X.X.csv`
- With Pearson: `pearson_Labsensor_10_SAGE++DAMC_..._delta_X.X.csv`

## Data Processing Pipeline
1. **Data Loading**: Load IBRL sensor data from text file
2. **Normalization**: StandardScaler normalization across all features
3. **Missingness Masking**: Create binary mask from NaN values
4. **Train/Eval Split**: Randomly select missing values for evaluation
5. **Imputation**: Apply continuous GNN model for imputation
6. **Evaluation**: Calculate MAE, MSE, MAPE metrics on eval set

## Features
- Support for multiple GNN architectures (SAGE, GAT, GCN, SAGE++DA, SAGE++DAC, SAGE++DAMC)
- Pearson correlation-based graph filtering with configurable thresholds
- Continuous streaming experiments with incremental updates
- Dynamic and static graph construction options
- Comprehensive evaluation metrics (MAE, MSE, MAPE)

## Notes
- The Labsensor dataset has 3 features compared to other datasets
- Window size of 24 epochs is recommended for this dataset
- Run time depends on the number of experiments and model complexity
- GPU acceleration can be enabled by modifying the device setting in `continuous.py`
