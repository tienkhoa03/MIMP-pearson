# Fast Testing Script for SAGE++DAMC experiments
# Optimized for faster execution with reduced data and iterations
# Usage: .\run_sage_damc_experiments_fast.ps1

# Run from the snapshot root (one level up from this scripts/ folder) so that
# MPIN-plus.py and ./exp_results/ resolve correctly.
Set-Location (Split-Path -Parent $PSScriptRoot)

Write-Host "========================================" -ForegroundColor Green
Write-Host "Starting SAGE++DAMC FAST TEST Experiments" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "FAST MODE Configuration:" -ForegroundColor Cyan
Write-Host "  - Datasets: ICU, Airquality" -ForegroundColor Cyan
Write-Host "  - Model: SAGE++DAMC" -ForegroundColor Cyan
Write-Host "  - Miss rates: 0.1, 0.3, 0.5" -ForegroundColor Cyan
Write-Host "  - Cases: With Pearson (delta=0.4), Without Pearson" -ForegroundColor Cyan
Write-Host "  - Epochs: 50 (reduced from 200)" -ForegroundColor Yellow
Write-Host "  - Iterations: 2 (reduced from 5)" -ForegroundColor Yellow
Write-Host "  - Data stream: 0.2 (use only 20% of data)" -ForegroundColor Yellow
Write-Host "  - k: 5 (reduced from 10)" -ForegroundColor Yellow
Write-Host ""

# Configuration for fast testing
$datasets = @("ICU", "Airquality")
$missRates = @(0.1, 0.3, 0.5)
$delta = 0.4

# OPTIMIZED PARAMETERS FOR SPEED
$epochs = 100          # Reduced from 200 -> 4x faster
$numIter = 1          # Reduced from 5 -> 2.5x faster
$stream = 0.2         # Use only 20% of data -> 5x faster
$k = 10                # Reduced from 10 -> faster graph construction
$base = "SAGE++DAMC"

Write-Host "Speed improvements:" -ForegroundColor Magenta
Write-Host "  - Epochs: 200 -> 50 (4x faster)" -ForegroundColor Magenta
Write-Host "  - Iterations: 5 -> 2 (2.5x faster)" -ForegroundColor Magenta
Write-Host "  - Data: 100% -> 20% (5x faster)" -ForegroundColor Magenta
Write-Host "  - KNN: 10 -> 5 (faster)" -ForegroundColor Magenta
Write-Host "  - Estimated speedup: ~40-50x faster!" -ForegroundColor Magenta
Write-Host ""

# Calculate total experiments
$totalExperiments = $datasets.Count * $missRates.Count * 2
$currentExperiment = 0
$experimentResults = @()

$globalStartTime = Get-Date

# ========================================
# CASE 1: WITH PEARSON FILTERING (delta=0.4)
# ========================================
Write-Host "########################################" -ForegroundColor Magenta
Write-Host "# CASE 1: WITH PEARSON (delta=$delta)    #" -ForegroundColor Magenta
Write-Host "########################################" -ForegroundColor Magenta
Write-Host ""

foreach ($dataset in $datasets) {
    foreach ($missRate in $missRates) {
        $currentExperiment++
        Write-Host "========================================" -ForegroundColor Yellow
        Write-Host "Experiment $currentExperiment/$totalExperiments" -ForegroundColor Yellow
        Write-Host "Dataset: $dataset | Miss Rate: $missRate | Pearson: YES (delta=$delta)" -ForegroundColor Yellow
        Write-Host "========================================" -ForegroundColor Yellow
        
        $startTime = Get-Date
        
        # Run the experiment with Pearson
        python MPIN-plus.py `
            --dataset $dataset `
            --k $k `
            --epochs $epochs `
            --stream $stream `
            --prefix fast_pearson `
            --num_of_iter $numIter `
            --base $base `
            --eval_ratio $missRate `
            --use_pearson true `
            --delta $delta
        
        $endTime = Get-Date
        $duration = $endTime - $startTime
        
        $experimentResults += [PSCustomObject]@{
            Experiment = $currentExperiment
            Dataset = $dataset
            MissRate = $missRate
            Pearson = "YES (delta=$delta)"
            Duration = "$($duration.TotalMinutes.ToString('F2')) min"
        }
        
        Write-Host ""
        Write-Host "Completed in $($duration.TotalMinutes.ToString('F2')) minutes" -ForegroundColor Green
        Write-Host ""
    }
}

# ========================================
# CASE 2: WITHOUT PEARSON FILTERING
# ========================================
Write-Host "########################################" -ForegroundColor Magenta
Write-Host "# CASE 2: WITHOUT PEARSON              #" -ForegroundColor Magenta
Write-Host "########################################" -ForegroundColor Magenta
Write-Host ""

foreach ($dataset in $datasets) {
    foreach ($missRate in $missRates) {
        $currentExperiment++
        Write-Host "========================================" -ForegroundColor Yellow
        Write-Host "Experiment $currentExperiment/$totalExperiments" -ForegroundColor Yellow
        Write-Host "Dataset: $dataset | Miss Rate: $missRate | Pearson: NO" -ForegroundColor Yellow
        Write-Host "========================================" -ForegroundColor Yellow
        
        $startTime = Get-Date
        
        # Run the experiment without Pearson
        python MPIN-plus.py `
            --dataset $dataset `
            --k $k `
            --epochs $epochs `
            --stream $stream `
            --prefix fast `
            --num_of_iter $numIter `
            --base $base `
            --eval_ratio $missRate `
            --use_pearson false
        
        $endTime = Get-Date
        $duration = $endTime - $startTime
        
        $experimentResults += [PSCustomObject]@{
            Experiment = $currentExperiment
            Dataset = $dataset
            MissRate = $missRate
            Pearson = "NO"
            Duration = "$($duration.TotalMinutes.ToString('F2')) min"
        }
        
        Write-Host ""
        Write-Host "Completed in $($duration.TotalMinutes.ToString('F2')) minutes" -ForegroundColor Green
        Write-Host ""
    }
}

$globalEndTime = Get-Date
$totalDuration = $globalEndTime - $globalStartTime

# ========================================
# SUMMARY
# ========================================
Write-Host ""
Write-Host "########################################" -ForegroundColor Green
Write-Host "#     ALL FAST TESTS COMPLETED!        #" -ForegroundColor Green
Write-Host "########################################" -ForegroundColor Green
Write-Host ""
Write-Host "Total time: $($totalDuration.TotalMinutes.ToString('F2')) minutes ($($totalDuration.TotalHours.ToString('F2')) hours)" -ForegroundColor Cyan
Write-Host ""

Write-Host "Experiment Summary:" -ForegroundColor Cyan
Write-Host "-------------------" -ForegroundColor Cyan
$experimentResults | Format-Table -AutoSize

Write-Host ""
Write-Host "Result files are in ./exp_results/ folder:" -ForegroundColor Cyan
Write-Host ""

# Check for generated files
Write-Host "WITH PEARSON (delta=$delta):" -ForegroundColor Yellow
foreach ($dataset in $datasets) {
    foreach ($missRate in $missRates) {
        $filename = "fast_pearson_${dataset}_${k}_${base}_incre_alone_window_2_epoch_${epochs}_eval_${missRate}_pearson_${delta}.csv"
        if (Test-Path "./exp_results/$filename") {
            Write-Host "  [OK] $filename" -ForegroundColor Green
        } else {
            Write-Host "  [MISSING] $filename" -ForegroundColor Red
        }
    }
}

Write-Host ""
Write-Host "WITHOUT PEARSON:" -ForegroundColor Yellow
foreach ($dataset in $datasets) {
    foreach ($missRate in $missRates) {
        $filename = "fast_${dataset}_${k}_${base}_incre_alone_window_2_epoch_${epochs}_eval_${missRate}.csv"
        if (Test-Path "./exp_results/$filename") {
            Write-Host "  [OK] $filename" -ForegroundColor Green
        } else {
            Write-Host "  [MISSING] $filename" -ForegroundColor Red
        }
    }
}

Write-Host ""
Write-Host "NOTE: This is a FAST TEST mode. For production results, use the full script." -ForegroundColor Yellow
Write-Host "Done!" -ForegroundColor Green
