param(
    [string]$CudaVersion = ''
)

Write-Host "Setting up Python 3.11 virtual environment 'venv311' and installing dependencies..."

if (-Not (Test-Path -Path .\venv311)) {
    Write-Host "Creating virtual environment using Python 3.11..."
    py -3.11 -m venv venv311
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment. Ensure Python 3.11 is installed and 'py -3.11' works."; exit 1
    }
}

if (-Not (Test-Path -Path .\venv311\Scripts\python.exe)) {
    Write-Host "Existing venv311 is missing python.exe; recreating it..."
    Remove-Item -Recurse -Force .\venv311
    py -3.11 -m venv venv311
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to recreate virtual environment. Ensure Python 3.11 is installed and 'py -3.11' works."; exit 1
    }
}

$venvPython = ".\venv311\Scripts\python.exe"

& $venvPython -c "import sys"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Existing venv311 is broken; recreating it from Python 3.11..."
    Remove-Item -Recurse -Force .\venv311
    py -3.11 -m venv venv311
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to recreate virtual environment. Ensure Python 3.11 is installed and 'py -3.11' works."; exit 1
    }
}

Write-Host "Upgrading pip, setuptools, wheel..."
& $venvPython -m pip install --upgrade pip wheel "setuptools<82"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to upgrade pip, setuptools, or wheel."; exit 1
}

if (-Not $CudaVersion -or $CudaVersion -eq '') {
    $choice = Read-Host "Choose CUDA for PyTorch (enter: 13.2, cpu). Leave empty for 13.2"
    if (-Not $choice) { $CudaVersion = '13.2' } else { $CudaVersion = $choice }
}

switch ($CudaVersion) {
    '13.2' { $index = 'https://download.pytorch.org/whl/cu132' }
    'cpu'  { $index = 'https://download.pytorch.org/whl/cpu' }
    default { Write-Host "Unrecognized CUDA option '$CudaVersion'. Using 13.2"; $index = 'https://download.pytorch.org/whl/cu132' }
}

Write-Host "Installing PyTorch + torchvision from $index ..."
& $venvPython -m pip install --upgrade --force-reinstall --no-cache-dir torch torchvision --index-url $index
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install PyTorch packages."; exit 1
}

& $venvPython -m pip install torch-geometric tensorboard pycorruptor
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install torch-geometric, tensorboard, or pycorruptor."; exit 1
}

$filteredRequirements = Join-Path $env:TEMP 'requirements311.filtered.txt'
Get-Content .\requirements311.txt | Where-Object { $_.Trim() -ne 'pypots==0.0.7' } | Set-Content $filteredRequirements

Write-Host "Installing packages from requirements311.txt..."
& $venvPython -m pip install --no-build-isolation -r $filteredRequirements
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install packages from requirements311.txt."; exit 1
}

Write-Host "Installing pypots without dependency resolution..."
& $venvPython -m pip install --no-deps pypots==0.0.7
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install pypots."; exit 1
}

Write-Host "If you need torch-geometric (PyG), install it separately. See https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html"
Write-Host "Done. Use '.\venv311\Scripts\python.exe' to run scripts with Python 3.11."