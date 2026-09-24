# src/ — гибридный сервис контроля качества DXA/ДРА

Пакет объединяет лучшее из трёх источников (base `dxa_qc`, `opus_solution`,
`fable_solution`). Полное обоснование выбора — в `../REPORT.md`.

## Модули

| модуль | назначение | откуда |
|---|---|---|
| `config.py` | единая конфигурация: таксономия, v3-источники, флаги, пороги | base + v3 |
| `preprocess.py` | пиксельный конвейер: modality/VOI LUT, костное окно, denoise, CLAHE | fable |
| `dicom_io.py` | чтение/дедуп/де-ид DICOM, определение области, масштаб пикселя | base (+ делегирование в `preprocess`) |
| `features.py` | геометрические признаки, вкл. `spine_midline_residual`, `art_*`, `atlas_*` | base + v3 |
| `atlas.py` | костный атлас-шаблон + артефакты вне кости (`art_*`, `atlas_*`) | новый |
| `metrics.py` | пороги `f1`/`prior`/`blend`, ДИ, метка организатора (без torch) | base + fable |
| `models/backbones.py` | реестр бэкбонов + `infer_backbone_from_weights` | base + fable |
| `models/hybrid_model.py` | мультизадачная сеть (region/quality/violation) | base |
| `models/losses.py` | masked BCE + Focal Loss | base |
| `models/tta.py` | безопасный TTA (малые повороты, без отражений) | base |
| `data/manifest.py` | построение манифеста, метки, `real_mask` | base |
| `data/dataset.py` | torch Dataset + аугментации + сэмплер | base |
| `data/synthetic.py` | синтетические артефакты v2 (только обучение) | fable |
| `stack.py` | гибридный стекер CNN + геометрия (карта v3) | base + v3 |
| `train.py` | честная CV: StratifiedGroupKFold, честная эпоха, Focal Loss | base + v3 |
| `calibrate.py` | подбор порогов по OOF -> `thresholds.json` | base |
| `evaluate.py` | BA / Macro-F1 / ROC-AUC с 95% ДИ | новый |
| `inference.py` | пакетный инференс DICOM -> отчёт по ТЗ | base (+ fix `use_cnn`) |
| `predict.py` | CLI пакетной обработки | base |

## Ключевые флаги (`config.py`)

| флаг | значение | смысл |
|---|---|---|
| `CRITERION_SOURCES` | spine_axis=fused, spine_artifacts=fused, spine_positioning=fused, femur_positioning=geo | карта v3 + укладка fused (atlas-признаки) |
| `THRESHOLD_MODE` | `f1` | воспроизводит подтверждённый 0.563; `blend` — устойчивая альтернатива |
| `QUALITY_LOSS` | `focal` | сильный дисбаланс (6–36 позитивов) |
| `PREPROCESS_MODE` | `bone` | костное окно + denoise + CLAHE |
| `ENSEMBLE_BACKBONES` | resnet18, resnet34 | усреднение вероятностей |
| `USE_COPIES_FEATURE` | `False` | аудит утечек (метаданные выгрузки) |
| `USE_STRATIFIED_GROUP` | `True` | стратификация фолдов по области |

## Пайплайн

```bash
python -m src.train --dataset-root <НД_для_обучения> --rebuild-manifest   # манифест + CV + финальная модель
python -m src.atlas                                                       # костный атлас-шаблон
python -m src.stack                                                       # гибридный стекер
python -m src.calibrate                                                   # пороги
python -m src.evaluate --thr blend --stacked                             # метрики + report.md
python -m src.predict --input <НД_для_обучения> --output outputs/results.csv
```

## Выходной формат (ТЗ)

`path_to_study, study_uid, image_uid, anatomical_region, quality_class,
violation_type, processing_status, time_of_processing`
(+ `quality_prob`, `violation_probs`). Любой сбой изображения →
`processing_status='Failure'`, батч продолжается.