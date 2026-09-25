# Наработки от Astra — гибридный сервис контроля качества DXA/ДРА

Самодостаточное решение для автоматической оценки качества DXA/ДРА-исследований
(поясничный отдел позвоночника + проксимальные отделы бедра) с отчётом в CSV/XLSX
по формату организатора.

Решение **объединяет** лучшее из двух базовых наработок и добавляет собственную
идею «кость + атлас-регистрация»:

| Источник | Что взято |
|---|---|
| `наработки от опуса` (v3) | конфигурация `CRITERION_SOURCES` / `CRITERION_FEATURES_OVERRIDE` (источник скора по критерию), fused-стекер, режим порога `f1`, `spine_axis` = угол средней линии + остаточная кривизна |
| `наработки от фейбла` | предобработка «костное окно + denoise + CLAHE», `infer_backbone_from_weights()`, синтетика артефактов v2, режим порога `blend`, ансамбль бэкбонов |
| **Astra (новое)** | изоляция кости + **атлас-регистрация** (`src/atlas.py`): шаблон кости строится БЕЗ меток; `art_*` — яркие объекты/резкие края вне кости; `atlas_*`/`align_*` — остаток/наклон к шаблону. Улучшает «предметы» и «укладку» |

## Структура

```
наработки от астры/
├── src/                 # самодостаточный пакет решения (единственный источник правды)
│   ├── config.py        # таксономия, пути, источники/признаки по критериям
│   ├── data/            # манифест, синтетика артефактов
│   ├── models/          # бэкбоны, определение архитектуры по весам
│   ├── preprocess.py    # костное окно + denoise + CLAHE
│   ├── features.py      # геометрия (spine_*/femur_*) + art_*/atlas_*/fatlas_*
│   ├── atlas.py         # построение шаблона и признаки атлас-регистрации
│   ├── train.py         # GroupKFold по study_uid, Focal/WeightedBCE, AMP
│   ├── stack.py         # гибридный стекер CNN + геометрия
│   ├── stack_clf.py     # классификаторы стекера (LogReg / LDA / balanced bagging)
│   ├── calibrate.py     # выбор порогов (f1/blend/prior/nested/mapped)
│   ├── evaluate.py      # метрики + organizer macro-F1 + CI
│   ├── inference.py     # инференс исследования (try/except -> Failure)
│   └── predict.py       # пакетный DICOM -> CSV/XLSX
├── scripts/             # эксперименты, абляции, диагностика
├── results/             # артефакты экспериментов (тексты, монтажи)
├── docs/report.md       # подробный сравнительный отчёт
├── artifacts/           # веса, пороги, стекеры, манифест (крупные файлы в .gitignore)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Быстрый старт

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
# torch/torchvision отдельно: CPU или CUDA (H200 -> cu124)
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Обучение (CV + финальные веса), затем стекер/калибровка/оценка:

```bash
python -m src.train --epochs 24 --folds 5
python -m src.atlas --region both        # костные шаблоны позвоночника и бедра
python -m src.stack && python -m src.calibrate && python -m src.evaluate --stacked
```

Пакетный инференс DICOM -> CSV/XLSX:

```bash
python -m src.predict --input /data/input --output /data/output/results.csv
```

## Выходной формат

Обязательные колонки (порядок фиксирован, см. `src/config.py:REPORT_COLUMNS`):

`path_to_study`, `study_uid`, `image_uid`, `anatomical_region`, `quality_class`,
`violation_type`, `processing_status`, `time_of_processing`.

* `anatomical_region` — «Поясничный отдел позвоночника» | «Проксимальный отдел бедра»;
* `quality_class` — 0/1 (годно/не годно);
* `violation_type` — мультилейбл, закрытый RU-список через `"; "`, пусто если нарушений нет;
* `processing_status` — `Success` | `Failure` (любая ошибка кадра не роняет батч).

## Ключевые флаги окружения

| Флаг | Значение | Смысл |
|---|---|---|
| `DXA_DATASET` | путь | корень набора `НД_для_обучения` (иначе ищется рядом с проектом) |
| `DXA_ARTIFACTS` | путь | каталог артефактов (по умолчанию `./artifacts`) |
| `DXA_FEATURE_MODE` | `new` / `base` | `base` — абляция «до/после» (базовые наборы, укладка бедра = geo) |

## Результаты (реальный набор организатора, 252 снимка)

Честная абляция полного конвейера (`scripts/run_ablation.py`, один 5-fold CV,
порог `mapped`), режим `base` (v3-базовые наборы) против `new`:

| критерий | AUC base | AUC new | F1 base | F1 new |
|---|---|---|---|---|
| spine_positioning | 0.758 | 0.857 | 0.600 | 0.667 |
| spine_axis | 0.826 | 0.826 | 0.500 | 0.500 |
| spine_artifacts | 0.753 | 0.859 | 0.500 | 0.634 |
| femur_positioning | 0.566 | 0.668 | 0.483 | 0.500 |
| femur_roi | 0.839 | 0.861 | 0.345 | 0.625 |

organizer macro-F1: **0.304 -> 0.460** (укладка 0.062->0.476, ось 0.438,
предметы 0.429->0.611, ROI 0.286->0.316).
`quality_class`: BA 0.566->0.710, macroF1 0.561->0.686, AUC 0.618->0.732.

Ключевой прирост даёт идея Astra: укладка позвоночника (AUC 0.758->0.857,
F1 0.600->0.667) и предметы (AUC 0.753->0.859, F1 0.500->0.634). Укладка бедра
улучшена отдельно (AUC 0.566->0.668) набором `femur_width_cm + femur_axis_angle
+ femur_bbox_h_ratio` в fused-режиме — признаки локальны для бедра, поэтому
остальные критерии не затрагиваются.

**Оговорка о протоколе.** Абсолютные числа зависят от протокола оценки
(число разбиений, режим порога). Устойчивый сигнал — дельта `base -> new` на
одном и том же протоколе; сравнение с числами из отчётов других решений
корректно только при совпадении протокола.

### Улучшение порогов (`mapped`, без переобучения)

Метрика организатора считается по OR-меткам, поэтому порог подобран per-criterion:
`prior` (по доле) для 4 критериев и `f1` для `femur_roi` (там `prior`
вырождается). На реальных данных (честно, порог с train): organizer macro-F1
**0.338 → 0.439**, `quality_class` BA **0.585 → 0.696**, macro-F1 **0.521 → 0.675**,
ROC-AUC 0.714 без изменений; ROI не ухудшен (0.286). Кластер-бутстрэп: прирост
macro-F1 +0.096 (P=0.99). Скрипт — `scripts/diag_threshold_final.py`,
подробности — раздел 8 `docs/report.md`.

### Улучшение источников `femur_roi` / `spine_artifacts` (v4)

Порог поднял метрику за счёт правила; следующий шаг — **источник скора** для самых
редких критериев (`scripts/exp_roi_source_real.py`, разбор
`results/v4_source_upgrade.md`):

```python
CRITERION_SOURCES = {
    "spine_positioning": "fused",
    "spine_axis": "fused",
    "spine_artifacts": "geo_bag",   # было fused: чистая геометрия art_*/atlas_* + bagging
    "femur_positioning": "fused",
    "femur_roi": "fused_bag",       # было cnn: гибрид + balanced bagging (7 позитивов!)
}
```

Суффикс источника (`_lda`, `_bag`) выбирает классификатор (`src/stack_clf.py`).

Seed-averaged (30 сидов × 5 фолдов `StratifiedGroupKFold` по `study_uid`, порог
только по train-части) base → v4: organizer macro-F1 **0.352 → 0.505** (+0.153,
30/30 сидов, Wilcoxon p<0.001), `quality_class` BA **0.663 → 0.705**, macro-F1
**0.649 → 0.682**, ROC-AUC **0.717 → 0.735**. Исторический `fold_id`
(`python -m src.evaluate --stacked`): macro-F1 **0.439 → 0.460**, BA **0.696 →
0.710**, macro-F1 **0.675 → 0.686**, ROC-AUC **0.714 → 0.732**; метки «укладка» и
«ось» без изменений, «предметы» 0.556→0.611, ROI 0.286→0.316. **Ни одна метрика
не ухудшилась ни на одном протоколе.**

## Инфраструктура

Цель — Linux + Docker + 2×NVIDIA H200, FP16/BF16, DDP/TorchScript.
`Dockerfile` собирается под CPU и GPU (ARG `TORCH_INDEX_URL`, `BASE_IMAGE`),
`docker-compose.yml` поднимает сервис с пробросом GPU и mini-PACS Orthanc.

## Подробности

Сравнительный отчёт и обоснование вклада каждого решения — `docs/report.md`.
