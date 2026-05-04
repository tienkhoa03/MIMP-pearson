# Continuous batch experiments for Airquality
# Cases:
# - base: SAGE++DAMC only
# - eval_ratio: 0.1, 0.3, 0.5
# - stream ratio: 0.1%, 1%, 10%, 100%
# - no Pearson
# - Pearson delta: 0.1..0.9
# Fixed window length: 2 (hours)
# Usage: .\run_airquality_experiments.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Push-Location $PSScriptRoot
try {
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Continuous Airquality Experiments" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green

    $dataset = "Airquality"
    $window = 2
    $bases = @("SAGE++DAMC")
    $evalRatios = @(0.1, 0.3, 0.5)
    $deltas = @(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    $streamRatios = @(0.001, 0.01, 0.1, 1.0)

    $epochs = 200
    $numIter = 5
    $k = 10
    $prefixNoPearson = "no_pearson"
    $prefixPearson = "pearson"

    $totalExperiments = ($bases.Count * $evalRatios.Count * $streamRatios.Count) + ($bases.Count * $evalRatios.Count * $deltas.Count * $streamRatios.Count)
    $current = 0
    $globalStart = Get-Date

    foreach ($base in $bases) {
        foreach ($eval in $evalRatios) {
            foreach ($stream in $streamRatios) {
                $current++
                Write-Host "[$current/$totalExperiments] dataset=$dataset base=$base eval=$eval pearson=false stream=$stream"

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
    }

    foreach ($base in $bases) {
        foreach ($eval in $evalRatios) {
            foreach ($delta in $deltas) {
                foreach ($stream in $streamRatios) {
                    $current++
                    Write-Host "[$current/$totalExperiments] dataset=$dataset base=$base eval=$eval pearson=true delta=$delta stream=$stream"

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
    }

    $globalEnd = Get-Date
    $duration = $globalEnd - $globalStart
    Write-Host "Completed $totalExperiments experiments in $($duration.TotalMinutes.ToString('F2')) minutes." -ForegroundColor Green
}
finally {
    Pop-Location
}
