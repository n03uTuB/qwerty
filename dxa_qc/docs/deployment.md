# Руководство по развёртыванию DXA-QC

## 1. Требования

- Docker 20.10+ (для GPU — Docker с поддержкой NVIDIA Container Toolkit);
- Linux/UNIX (для `run.sh`) либо Windows (команды `docker` напрямую);
- для GPU-варианта: драйвер NVIDIA + CUDA 12.4 runtime.

## 2. Сборка образа

### CPU

```bash
./run.sh build
# или
docker build -t dxa-qc .
```

### GPU (CUDA 12.4)

```bash
./run.sh build-gpu
# или
docker build \
  --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
  -t dxa-qc:gpu .
```

## 3. Запуск сервиса

```bash
./run.sh api          # CPU, http://localhost:8000
./run.sh api-gpu      # GPU (--gpus all)
```

Проверка:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/model/info
```

## 4. Пакетная обработка на целевом оборудовании

```bash
./run.sh predict /data/dicom /data/out/results.csv
```

Каталог с входными данными монтируется только для чтения; результаты
записываются в отдельный каталог.

## 5. Масштабирование

- решение не имеет состояния (stateless) — можно запускать несколько
  контейнеров за балансировщиком;
- на конфигурации организатора (2 × H200) рекомендуется запускать по одному
  процессу на GPU (`--gpus '"device=0"'` и `--gpus '"device=1"'`) и распределять
  исследования между ними;
- узкое место инференса — декодирование DICOM и прогон сети; при большом потоке
  целесообразна очередь заданий (например, Celery/RQ) поверх API.

## 6. Переменные окружения

| Переменная | Назначение | По умолчанию |
|---|---|---|
| `OMP_NUM_THREADS` | число потоков CPU | 8 |
| `MKL_NUM_THREADS` | число потоков MKL | 8 |

## 7. Обновление модели

1. положите новые веса в `artifacts/best_model.pt` (и `fold_k.pt` при ансамбле);
2. при необходимости обновите `artifacts/thresholds.json`;
3. пересоберите образ: `./run.sh build`.

## 8. Диагностика

| Проблема | Решение |
|---|---|
| `Не найдено ни одного файла весов` | обучите модель (`python -m src.train`) или положите веса в `artifacts/` |
| контейнер не видит GPU | установите NVIDIA Container Toolkit, используйте `--gpus all` |
| долгий первый запуск | прогревается загрузка torch; при повторных запусках быстрее |
| ошибки чтения DICOM | проверьте целостность файлов; строки с ошибками помечаются `Failure` |
