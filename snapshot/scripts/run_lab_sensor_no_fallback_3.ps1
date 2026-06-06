# Labsensor experiments — Pearson-filtered graph WITHOUT KNN fallback
# Sweep   : delta (Pearson threshold) from 0.9 down to 0.1
# Config  : dataset=Labsensor, window=30, stream=0.01, eval=[0.1,0.3,0.5]
#           use_pearson=true, fallback_knn=false (empty graph when no correlated pairs)
# Model   : SAGE++DAMC (change $base below if a different base is needed)
# Usage   : .\run_lab_sensor_no_fallback.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $pythonExe = Join-Path $PSScriptRoot "..\..\venv311\Scripts\python.exe"
    if (-Not (Test-Path $pythonExe)) {
        $pythonExe = "python"
    }

    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Labsensor Pearson sweep (NO KNN fallback)" -ForegroundColor Green
    Write-Host "  delta 0.9 -> 0.1 | eval 0.1/0.3/0.5" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Green

    $cudaStatus = & $pythonExe -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"
    Write-Host "Using $cudaStatus via $pythonExe" -ForegroundColor Green

    $dataset    = "Labsensor"
    $window     = 30
    $stream     = 0.01
    $base       = "SAGE++DAMC"
    $prefix     = "no_fallback"
    $evalRatios = @(0.5)
    $deltas     = @(0.5, 0.4, 0.3, 0.2, 0.1)

    $totalExp    = $deltas.Count * $evalRatios.Count
    $current     = 0
    $globalStart = Get-Date

    foreach ($delta in $deltas) {
        foreach ($eval in $evalRatios) {
            $current++
            $startTime = Get-Date
            Write-Host ""
            Write-Host "[$current/$totalExp] delta=$delta  eval=$eval  fallback_knn=false" -ForegroundColor Yellow

            & $pythonExe MPIN-plus.py `
                --dataset      $dataset `
                --window       $window `
                --stream       $stream `
                --base         $base `
                --prefix       $prefix `
                --eval_ratio   $eval `
                --use_pearson  true `
                --delta        $delta `
                --fallback_knn false

            $dur = (Get-Date) - $startTime
            Write-Host "  done in $($dur.TotalMinutes.ToString('F2')) min" -ForegroundColor Green
        }
    }

    $totalDur = (Get-Date) - $globalStart
    Write-Host ""
    Write-Host "All $totalExp experiments completed in $($totalDur.TotalMinutes.ToString('F2')) min." -ForegroundColor Green
}
finally {
    Pop-Location
}
