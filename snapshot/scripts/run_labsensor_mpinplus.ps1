# Labsensor experiments — MPIN-plus GNN baselines
# Bases  : SAGE (mpin), GAT (mimp-gatv2), SAGE++DA (mimp-sage++da)
# Config : dataset=Labsensor, window=30, stream=1, eval=[0.1,0.3,0.5], no Pearson
# Usage  : .\run_labsensor_mpinplus.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $pythonExe = Join-Path $PSScriptRoot "..\..\venv311\Scripts\python.exe"
    if (-Not (Test-Path $pythonExe)) {
        $pythonExe = "python"
    }

    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Labsensor MPIN-plus Experiments" -ForegroundColor Green
    Write-Host "  SAGE (mpin) | GAT (mimp-gatv2) | SAGE++DA (mimp-sage++da)" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Green

    $cudaStatus = & $pythonExe -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"
    Write-Host "Using $cudaStatus via $pythonExe" -ForegroundColor Green

    $dataset    = "Labsensor"
    $window     = 30
    $stream     = 0.01
    $evalRatios = @(0.1, 0.3, 0.5)
    $epochs     = 200
    $numIter    = 5
    $k          = 10
    $prefix     = "no_pearson"

    # SAGE  → mpin (plain GraphSAGE)
    # GAT   → mimp(gatv2)
    # SAGE++DA → mimp(sage++da)
    $bases = @("SAGE", "GAT", "SAGE++DA")

    $totalExp    = $bases.Count * $evalRatios.Count
    $current     = 0
    $globalStart = Get-Date

    foreach ($base in $bases) {
        foreach ($eval in $evalRatios) {
            $current++
            $startTime = Get-Date
            Write-Host ""
            Write-Host "[$current/$totalExp] base=$base  eval=$eval" -ForegroundColor Yellow

            & $pythonExe MPIN-plus.py `
                --dataset     $dataset `
                --window      $window `
                --k           $k `
                --epochs      $epochs `
                --prefix      $prefix `
                --num_of_iter $numIter `
                --base        $base `
                --eval_ratio  $eval `
                --stream      $stream `
                --use_pearson false

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
