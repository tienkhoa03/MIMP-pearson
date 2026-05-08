# Continuous batch experiments for Labsensor (Intel Berkeley Research Lab)
# Sensor data with 3 features: temperature, humidity, light
# Fixed window length: 24 (epochs)
# Usage: .\run_labsensor_experiments.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Push-Location $PSScriptRoot
try {
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Continuous Labsensor Experiments" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green

    $dataset = "Labsensor"
    $window = 24
    $bases = @("SAGE++DAMC")
    $evalRatios = @(0.1, 0.3, 0.5)
    $deltas = @(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    $epochs = 200
    $numIter = 1
    $k = 10
    $stream = 1.0
    $prefixNoPearson = "no_pearson"
    $prefixPearson = "pearson"

    $totalExperiments = ($bases.Count * $evalRatios.Count) + ($bases.Count * $evalRatios.Count * $deltas.Count)
    $current = 0
    $globalStart = Get-Date

    # Run experiments without Pearson filtering
    foreach ($base in $bases) {
        foreach ($eval in $evalRatios) {
            $current++
            Write-Host "[$current/$totalExperiments] dataset=$dataset base=$base eval=$eval pearson=false" -ForegroundColor Cyan

            python continuous.py `
                --dataset $dataset `
                --window $window `
                --k $k `
                --epochs $epochs `
                --prefix $prefixNoPearson `
                --num_of_iter $numIter `
                --base $base `
                --eval_ratio $eval `
                --stream $stream `
                --use_pearson false
        }
    }

    # Run experiments with Pearson filtering
    foreach ($base in $bases) {
        foreach ($eval in $evalRatios) {
            foreach ($delta in $deltas) {
                $current++
                Write-Host "[$current/$totalExperiments] dataset=$dataset base=$base eval=$eval pearson=true delta=$delta" -ForegroundColor Cyan

                python continuous.py `
                    --dataset $dataset `
                    --window $window `
                    --k $k `
                    --epochs $epochs `
                    --prefix $prefixPearson `
                    --num_of_iter $numIter `
                    --base $base `
                    --eval_ratio $eval `
                    --stream $stream `
                    --use_pearson true `
                    --delta $delta
            }
        }
    }

    $globalEnd = Get-Date
    $elapsed = ($globalEnd - $globalStart).TotalSeconds
    Write-Host "Completed $totalExperiments experiments in $([Math]::Round($elapsed/60, 2)) minutes." -ForegroundColor Green
}
finally {
    Pop-Location
}
