# Labsensor experiments — traditional / NN baselines
# Methods: mean, KNN, MICE, MF  (via trad.py)
#          fp                    (via FP.py)
#          brits, saits          (via nn.py)
# Config : dataset=Labsensor, window=30, stream=1, eval=[0.1,0.3,0.5]
# NOTE   : trad.py / FP.py / nn.py must have Labsensor dataset support.
# Usage  : .\run_labsensor_trad.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Push-Location $PSScriptRoot
try {
    $pythonExe = Join-Path $PSScriptRoot "..\venv311\Scripts\python.exe"
    if (-Not (Test-Path $pythonExe)) {
        $pythonExe = "python"
    }

    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Labsensor Traditional Baseline Experiments" -ForegroundColor Green
    Write-Host "  mean | KNN | MICE | MF | FP | BRITS | SAITS" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Green

    $cudaStatus = & $pythonExe -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"
    Write-Host "Using $cudaStatus via $pythonExe" -ForegroundColor Green

    $dataset    = "Labsensor"
    $window     = 30
    $stream     = 0.01
    $evalRatios = @(0.1, 0.3, 0.5)
    $k          = 10
    $prefix     = "no_pearson"

    $tradMethods = @("mean", "KNN", "MICE", "MF-mf")
    $nnMethods   = @("brits")

    $totalExp    = ($tradMethods.Count + 1 + $nnMethods.Count) * $evalRatios.Count
    $current     = 0
    $globalStart = Get-Date

    # ----------------------------------------
    # trad.py  :  mean, KNN, MICE, MF
    # ----------------------------------------
    Write-Host ""
    Write-Host "--- trad.py : mean / KNN / MICE / MF ---" -ForegroundColor Magenta

    # ----------------------------------------
    # nn.py  :  BRITS, SAITS
    # ----------------------------------------
    Write-Host ""
    Write-Host "--- nn.py : BRITS / SAITS ---" -ForegroundColor Magenta

    foreach ($nnMethod in $nnMethods) {
        foreach ($eval in $evalRatios) {
            $current++
            $startTime = Get-Date
            Write-Host ""
            Write-Host "[$current/$totalExp] method=$nnMethod  eval=$eval" -ForegroundColor Yellow

            & $pythonExe nn.py `
                --dataset    $dataset `
                --window     $window `
                --prefix     $prefix `
                --method     $nnMethod `
                --eval_ratio $eval `
                --stream     $stream

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
