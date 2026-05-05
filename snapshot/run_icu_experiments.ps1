# Snapshot batch experiments for ICU
# Cases:
# - base: SAGE++DAC, SAGE++DAMC
# - eval_ratio: 0.1, 0.3, 0.5
# - no Pearson
# - Pearson delta: 0.1..0.9
# Fixed window length: 4
# Usage: .\run_icu_experiments.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:128"

Push-Location $PSScriptRoot
try {
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Snapshot ICU Experiments" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green

    $dataset = "ICU"
    $window = 4
    $bases = @("SAGE++DAMC")
    $evalRatios = @(0.3)
    $deltas = @(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    $epochs = 200
    $numIter = 5
    $k = 10
    $stream = 1

    $prefixNoPearson = "no_pearson"
    $prefixPearson = "pearson"

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

    foreach ($base in $bases) {
        foreach ($eval in $evalRatios) {
            foreach ($delta in $deltas) {
                $current++
                Write-Host "[$current/$totalExperiments] dataset=$dataset base=$base eval=$eval pearson=true delta=$delta"

                python MPIN-plus.py `
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
    $duration = $globalEnd - $globalStart
    Write-Host "Completed $totalExperiments experiments in $($duration.TotalMinutes.ToString('F2')) minutes." -ForegroundColor Green
}
finally {
    Pop-Location
}
