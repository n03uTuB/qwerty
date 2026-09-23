# syntax=docker/dockerfile:1
# Образ сервиса контроля качества денситометрии DXA-QC.
#
# Сборка CPU:  docker build -t dxa-qc .
# Сборка GPU:  docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
#                          --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 \
#                          -t dxa-qc:gpu .
#
# Целевая конфигурация организатора — 2 × NVIDIA H200 (141 GB VRAM, Hopper).
# H200 требует CUDA 12.x (cu124 подходит) и NVIDIA Container Toolkit на хосте
# (см. docker-compose.yml, секция deploy.resources.reservations.devices).
FROM python:3.10-slim AS base

ARG BASE_IMAGE=python:3.10-slim
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    # для NVIDIA Container Toolkit (проброс GPU в контейнер)
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# системные библиотеки для opencv/pydicom/albumentations
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) torch/torchvision под выбранный бэкенд (CPU или CUDA)
RUN pip install --upgrade pip && \
    pip install torch==2.14.0 torchvision==0.29.0 --index-url ${TORCH_INDEX_URL}

# 2) остальные зависимости (зафиксированы)
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3) исходный код и веса
COPY src ./src
COPY artifacts ./artifacts

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# по умолчанию — API-сервис
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
