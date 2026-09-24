# syntax=docker/dockerfile:1
# =============================================================================
# Гибридный сервис контроля качества DXA/ДРА (пакет src/).
#
# Сборка CPU:  docker build -t dxa-qc-hybrid .
# Сборка GPU:  docker build \
#                --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
#                --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 \
#                -t dxa-qc-hybrid:gpu .
#
# Целевая конфигурация организатора — 2 x NVIDIA H200 (141 GB, Hopper):
# H200 требует CUDA 12.x (cu124) и NVIDIA Container Toolkit на хосте
# (см. docker-compose.yml, секция deploy.resources.reservations.devices).
# =============================================================================
ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    # для NVIDIA Container Toolkit (проброс GPU в контейнер)
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# системные библиотеки для opencv-headless / pydicom / Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) torch/torchvision под выбранный бэкенд (CPU или CUDA)
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url ${TORCH_INDEX_URL}

# 2) остальные зависимости (зафиксированы в requirements.txt)
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3) исходный код и артефакты (веса, пороги, стекеры — если уже обучены)
COPY src ./src
COPY artifacts ./artifacts

# каталоги ввода/вывода монтируются томами (см. docker-compose.yml)
RUN mkdir -p /data/input /data/output

# healthcheck: пакет и модель импортируются, ядро видит GPU (если есть)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import src.inference, torch; print('ok', torch.cuda.is_available())" || exit 1

# по умолчанию — пакетный инференс DICOM -> CSV/XLSX (пороги берутся из artifacts/)
ENTRYPOINT ["python", "-m"]
CMD ["src.predict", "--input", "/data/input", "--output", "/data/output/results.csv"]