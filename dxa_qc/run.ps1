# Вспомогательный скрипт для Windows (аналог run.sh).
# Требуемый по ТЗ Linux/UNIX-скрипт — run.sh.
param(
    [Parameter(Position=0)][string]$Command = "help",
    [Parameter(Position=1)][string]$Input,
    [Parameter(Position=2)][string]$Output
)
$ErrorActionPreference = "Stop"
$Image = if ($env:IMAGE) { $env:IMAGE } else { "dxa-qc" }
$ImageGpu = if ($env:IMAGE_GPU) { $env:IMAGE_GPU } else { "dxa-qc:gpu" }
$Root = $PSScriptRoot

switch ($Command) {
    "build"     { docker build -t $Image $Root }
    "build-gpu" {
        docker build --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 `
            --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 `
            -t $ImageGpu $Root
    }
    "api"       { docker run --rm -p 8000:8000 $Image }
    "api-gpu"   { docker run --rm --gpus all -p 8000:8000 $ImageGpu }
    "predict"   {
        if (-not $Input -or -not $Output) { throw "Укажите: predict <входной каталог> <выходной csv>" }
        $outDir = Split-Path -Parent (Resolve-Path -LiteralPath (New-Item -ItemType File -Force -Path $Output)).FullName
        $inFull = (Resolve-Path -LiteralPath $Input).Path
        docker run --rm -v "${inFull}:/data/input:ro" -v "${outDir}:/data/output" $Image `
            python -m src.predict --input /data/input --output "/data/output/$(Split-Path -Leaf $Output)"
    }
    default {
        Write-Host "Использование: .\run.ps1 {build|build-gpu|api|api-gpu|predict <in> <out.csv>}"
    }
}
