param(
    [string]$CudaVersion = "12.1"
)

$ErrorActionPreference = "Stop"
$VENV_DIR = Join-Path $PSScriptRoot "awod_cuda"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "AWOD  CUDA 训练环境设置" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "GPU: NVIDIA GeForce RTX 4070 Laptop GPU" -ForegroundColor Green
Write-Host "CUDA: $CudaVersion" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan

if (Test-Path $VENV_DIR) {
    Write-Host "[SKIP] 虚拟环境已存在: $VENV_DIR" -ForegroundColor Yellow
} else {
    Write-Host "[1/3] 创建虚拟环境..." -ForegroundColor Cyan
    python -m venv $VENV_DIR
}

$ACTIVATE = Join-Path $VENV_DIR "Scripts\Activate.ps1"
Write-Host "[2/3] 激活环境并安装 PyTorch CUDA 版本..." -ForegroundColor Cyan
& $ACTIVATE

pip install --upgrade pip -q

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu$($CudaVersion.Replace('.','')) -q

Write-Host "[3/3] 安装项目依赖..." -ForegroundColor Cyan
$REQ = Join-Path $PSScriptRoot "requirements.txt"
pip install -r $REQ -q

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "环境安装完成!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "使用方法:" -ForegroundColor Yellow
Write-Host "  1. 激活环境: .\awod_cuda\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  2. 开始训练: python train_foggy.py" -ForegroundColor White
Write-Host "  3. 自定义参数: python train_foggy.py --epochs 100 --batch 16" -ForegroundColor White
Write-Host ""
