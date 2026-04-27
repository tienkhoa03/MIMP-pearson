# Script to run SAGE++DAMC experiments with and without Pearson filtering
# Datasets: ICU, Airquality
# Miss rates: 0.1, 0.3, 0.5
# Cases: With Pearson (delta=0.4) and Without Pearson
# Total: 2 datasets × 2 cases × 3 miss rates = 12 experiments
# Usage: .\run_sage_damc_experiments.ps1

Write-Host "========================================" -ForegroundColor Green
Write-Host "Starting SAGE++DAMC Experiments" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  - Datasets: ICU, Airquality" -ForegroundColor Cyan
Write-Host "  - Model: SAGE++DAMC" -ForegroundColor Cyan
Write-Host "  - Miss rates: 0.1, 0.3, 0.5" -ForegroundColor Cyan
Write-Host "  - Cases: With Pearson (delta=0.4), Without Pearson" -ForegroundColor Cyan
Write-Host "  - Epochs: 200, Iterations: 5, k: 10" -ForegroundColor Cyan
Write-Host ""

# Configuration
$datasets = @("ICU", "Airquality")
$missRates = @(0.1, 0.3, 0.5)
$delta = 0.4
$epochs = 200
$numIter = 5
$k = 10
$base = "SAGE++DAMC"

# Calculate total experiments
$totalExperiments = $datasets.Count * $missRates.Count * 2  # 2 for with/without Pearson
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
            --prefix khoa_pearson `
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
            --prefix khoa `
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
Write-Host "#          ALL EXPERIMENTS COMPLETED   #" -ForegroundColor Green
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
        $filename = "khoa_pearson_${dataset}_${k}_${base}_incre_alone_window_2_epoch_${epochs}_eval_${missRate}_pearson_${delta}.csv"
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
        $filename = "khoa_${dataset}_${k}_${base}_incre_alone_window_2_epoch_${epochs}_eval_${missRate}.csv"
        if (Test-Path "./exp_results/$filename") {
            Write-Host "  [OK] $filename" -ForegroundColor Green
        } else {
            Write-Host "  [MISSING] $filename" -ForegroundColor Red
        }
    }
}

Write-Host ""
Write-Host "Done!" -ForegroundColor Green
