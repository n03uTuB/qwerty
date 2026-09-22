#!/usr/bin/env bash
# Скрипт сборки и запуска контейнеризированного сервиса DXA-QC (Linux/UNIX).
#
# Примеры:
#   ./run.sh build                 # собрать CPU-образ
#   ./run.sh build-gpu             # собрать GPU-образ (CUDA)
#   ./run.sh api                   # запустить API на :8000
#   ./run.sh predict /data/in out/results.csv   # пакетная обработка каталога
#   ./run.sh shell                 # интерактивная оболочка в контейнере
set -euo pipefail

IMAGE="${IMAGE:-dxa-qc}"
IMAGE_GPU="${IMAGE_GPU:-dxa-qc:gpu}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

build() {
    docker build -t "${IMAGE}" "${PROJECT_DIR}"
}

build_gpu() {
    docker build \
        --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 \
        --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
        -t "${IMAGE_GPU}" "${PROJECT_DIR}"
}

api() {
    docker run --rm -p 8000:8000 "${IMAGE}"
}

api_gpu() {
    docker run --rm --gpus all -p 8000:8000 "${IMAGE_GPU}"
}

predict() {
    local input_dir="$1"
    local output_csv="$2"
    local out_dir
    out_dir="$(cd "$(dirname "${output_csv}")" && pwd)"
    docker run --rm \
        -v "${input_dir}:/data/input:ro" \
        -v "${out_dir}:/data/output" \
        "${IMAGE}" \
        python -m src.predict --input /data/input \
            --output "/data/output/$(basename "${output_csv}")"
}

shell() {
    docker run --rm -it -v "${PROJECT_DIR}:/app" "${IMAGE}" bash
}

case "${1:-}" in
    build)       build ;;
    build-gpu)   build_gpu ;;
    api)         api ;;
    api-gpu)     api_gpu ;;
    predict)     shift; predict "$@" ;;
    shell)       shell ;;
    *)
        echo "Использование: $0 {build|build-gpu|api|api-gpu|predict <in> <out.csv>|shell}"
        exit 1
        ;;
esac
