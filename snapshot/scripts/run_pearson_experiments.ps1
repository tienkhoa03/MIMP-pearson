# Script to run experiments with different Pearson delta thresholds
# Usage: .\run_pearson_experiments.ps1

# Run from the snapshot root (one level up from this scripts/ folder) so that
# MPIN-plus.py and ./exp_results/ resolve correctly.
Set-Location (Split-Path -Parent $PSScriptRoot)

Write-Host "Starting Pearson delta experiments..." -ForegroundColor Green
Write-Host "Dataset: Wifi, k=10, epochs=200, iterations=5" -ForegroundColor Cyan
Write-Host ""

# Array of delta values to test
$deltas = @(0.4, 0.5)
$k = @(11, 12, 13, 14, 15)
$eval_ratios = @(0.1, 0.3, 0.5)

$totalExperiments = $deltas.Count
$currentExperiment = 0

foreach ($delta in $deltas) {
    foreach ($k in $k) {
        foreach ($eval_ratios in $eval_ratios) {
    $currentExperiment++
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host "Experiment $currentExperiment/$totalExperiments : Delta = $delta" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow
    
    $startTime = Get-Date
    
    # Run the experiment
    python MPIN-plus.py `
        --dataset Airquality `
        --k $k `
        --epochs 200 `
        --prefix pearson `
        --num_of_iter 5 `
        --base SAGE++DAMC `
        --use_pearson true `
        --eval_ratio $eval_ratios `
        --window 2 `
        --stream 1 `
        --delta $delta
    
    $endTime = Get-Date
    $duration = $endTime - $startTime
    
    Write-Host ""
    Write-Host "Completed delta=$delta in $($duration.TotalMinutes.ToString('F2')) minutes" -ForegroundColor Green
    Write-Host ""
    }
    }   
}

Write-Host "========================================" -ForegroundColor Green
Write-Host "All experiments completed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Result files are in ./exp_results/ folder:" -ForegroundColor Cyan

# List the generated files
foreach ($delta in $deltas) {
    $filename = "pearson_$delta.csv"
    # if (Test-Path "./exp_results/$filename") {
        Write-Host "  [OK] $filename" -ForegroundColor Green
    # } else {
    #     Write-Host "  [MISSING] $filename" -ForegroundColor Red
    # }
}
