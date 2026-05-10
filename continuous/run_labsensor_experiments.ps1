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
    $window = 30
    $bases = @("SAGE++DAMC")
    $increModes = @("alone", "data", "transfer", "data+state+transfer")
    $evalRatios = @(0.3)
    $deltas = @(0.8)
    $streams = @(0.1, 0.5, 1.0)

    $epochs = 200
    $numIter = 5
    $k = 10
    $prefixNoPearson = "no_pearson"
    $prefixPearson = "pearson"

    $expsPerStream = ($bases.Count * $increModes.Count * $evalRatios.Count) + ($bases.Count * $increModes.Count * $evalRatios.Count * $deltas.Count)
    $totalExperiments = $expsPerStream * $streams.Count
    $current = 0
    $globalStart = Get-Date

    # Run experiments for each stream value
    # foreach ($streamVal in $streams) {
    #     Write-Host ""
    #     Write-Host "========== Starting experiments with stream=$streamVal ==========" -ForegroundColor Yellow
    #     Write-Host ""

    #     # Run experiments without Pearson filtering
    #     foreach ($base in $bases) {
    #         foreach ($increMode in $increModes) {
    #             foreach ($eval in $evalRatios) {
    #                 $current++
    #                 Write-Host "[$current/$totalExperiments] stream=$streamVal dataset=$dataset base=$base mode=$increMode eval=$eval pearson=false" -ForegroundColor Cyan

    #                 python continuous.py `
    #                     --dataset $dataset `
    #                     --window $window `
    #                     --k $k `
    #                     --epochs $epochs `
    #                     --prefix $prefixNoPearson `
    #                     --num_of_iter $numIter `
    #                     --base $base `
    #                     --incre_mode $increMode `
    #                     --eval_ratio $eval `
    #                     --stream $streamVal `
    #                     --use_pearson false
    #             }
    #         }
    #     }

        # Run experiments with Pearson filtering
        foreach ($base in $bases) {
            foreach ($increMode in $increModes) {
                foreach ($eval in $evalRatios) {
                    foreach ($delta in $deltas) {
                        $current++
                        Write-Host "[$current/$totalExperiments] stream=$streamVal dataset=$dataset base=$base mode=$increMode eval=$eval pearson=true delta=$delta" -ForegroundColor Cyan

                        python continuous.py `
                            --dataset $dataset `
                            --window $window `
                            --k $k `
                            --epochs $epochs `
                            --prefix $prefixPearson `
                            --num_of_iter $numIter `
                            --base $base `
                            --incre_mode $increMode `
                            --eval_ratio $eval `
                            --stream $streamVal `
                            --use_pearson true `
                            --delta $delta
                    }
                }
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
