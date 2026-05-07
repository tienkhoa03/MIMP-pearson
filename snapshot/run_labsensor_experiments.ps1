# Snapshot batch experiments for Labsensor
# Cases:
# - base: SAGE++DAC, SAGE++DAMC
# - eval_ratio: 0.1, 0.3, 0.5
# - no Pearson
# - Pearson delta: 0.1..0.9
# Fixed window length: 30
# Usage: .\run_labsensor_experiments.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Push-Location $PSScriptRoot
try {
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Snapshot Labsensor Experiments" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green

    $dataset = "Labsensor"
    $window = 30
    $bases = @("SAGE++DAC", "SAGE++DAMC")
    $evalRatios = @(0.1, 0.3, 0.5)
    $deltas = @(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    $stream = 0.01

    $totalExperiments = ($bases.Count * $evalRatios.Count) + ($bases.Count * $evalRatios.Count * $deltas.Count)
    $current = 0
    $globalStart = Get-Date

    foreach ($base in $bases) {
        foreach ($eval in $evalRatios) {
            $current++
            Write-Host "[$current/$totalExperiments] dataset=$dataset base=$base eval=$eval pearson=false"

            python MPIN-plus.py `
                --dataset $dataset `
                --window $window `
                --base $base `
                --eval_ratio $eval `
                --stream $stream `
                --use_pearson false
        }
    }

    foreach ($base in $bases) {
        foreach ($eval in $evalRatios) {
            foreach ($delta in $deltas) {
                $current++
                Write-Host "[$current/$totalExperiments] dataset=$dataset base=$base eval=$eval pearson=true delta=$delta"

                python MPIN-plus.py `
                    --dataset $dataset `
                    --window $window `
                    --base $base `
                    --eval_ratio $eval `
                    --stream $stream `
                    --use_pearson true `
                    --delta $delta
            }
        }
    }

    $globalEnd = Get-Date
    $duration = $globalEnd - $globalStart
    Write-Host "Completed $totalExperiments experiments in $($duration.TotalMinutes.ToString('F2')) minutes." -ForegroundColor Green
}
finally {
    Pop-Location
}