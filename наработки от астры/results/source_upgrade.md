# v4/v5: усиление источников скора по критериям

Дата: 2026-09-25. База сравнения — коммит `d256907` (organizer macro-F1 0.439
на историческом `fold_id`-протоколе).

Скрипт: `scripts/exp_roi_source_real.py` (честный замер на боевом контуре
`src.stack` + функции `src.evaluate`). Сырой лог: `results/roi_source_real.txt`.

## Протоколы

* **`fold_id`** — исторический боевой: 5 фолдов обучения CNN (`fold_id.npy`),
  порог только по train-части (`train_violation.npy`), метрика по val, пул по
  фолдам (совпадает с `python -m src.evaluate --stacked`).
* **`seeds`** — `StratifiedGroupKFold(5)` по `study_uid` × 30 сидов; порог только
  по train-части фолда; метрика — среднее по фолдам и сидам (устойчивее к
  одному разбиению).

Масштаб порога — `cnn` (по train-пробам CNN, исторический harness). Все замеры
только по реальным снимкам (252).

## Классификатор как часть источника (`src/stack_clf.py`)

Источник критерия (`config.CRITERION_SOURCES`) кодирует и входы, и модель:

| суффикс | модель | когда полезен |
|---|---|---|
| без суффикса (`fused`/`geo`) | LogReg | вариант v3 |
| `_lda` | LDA со shrinkage | 6–36 позитивов |
| `_bag` | balanced bagging (EasyEnsemble) | очень редкие классы |

Класс `BalancedBag` вынесен в отдельный модуль `src/stack_clf.py`, чтобы
корректно сериализоваться в joblib (`src.stack` запускается как `python -m
src.stack`, т.е. как `__main__`).

## v4: `femur_roi` и `spine_artifacts`

### `femur_roi`: `cnn` → `fused_bag` (крупнейший единичный прирост)

7 позитивов из 150; при источнике `cnn` и пороге `f1` рабочая точка вырождается
(seed-averaged F1 ≈ 0.005). Гибрид `[cnn_prob | femur_height_cm, femur_bone_ratio,
femur_margin_min_cm]` с balanced bagging даёт F1 ≈ 0.56.

Seed-averaged, 30 сидов (порог `mapped`, ROI-режим `f1`, масштаб `cnn`):

| источник ROI | macro | BA | qF1 | qAUC | ROI F1 |
|---|---|---|---|---|---|
| `cnn` (база) | 0.352 | 0.663 | 0.649 | 0.717 | 0.005 |
| `fused` (LogReg) | 0.484 | 0.691 | 0.670 | 0.726 | 0.531 |
| `fused_bag` | 0.490 | 0.695 | 0.673 | 0.731 | 0.559 |
| `geo_lda` | 0.491 | 0.691 | 0.676 | 0.725 | 0.561 |
| `geo_bag` | 0.433 | 0.684 | 0.664 | 0.720 | 0.324 |

Парно по 30 сидам против `cnn`: `fused` Δmacro +0.132 (30/30), `fused_bag`
+0.138 (30/30), `geo_lda` +0.139 (30/30); по BA/qF1/qAUC — тоже 30/30, p<0.001.

На историческом `fold_id`: `fused_bag` — единственный вариант, где **все**
метрики растут (macro 0.439→0.446, BA 0.696→0.700, qF1 0.675→0.677,
qAUC 0.714→0.728); `geo_lda` даёт больший macro (+0.095, P=0.97), но слегка
проседает BA (0.693). Поэтому выбран `fused_bag`.

### `spine_artifacts`: `fused` → `geo_bag`

Чистая геометрия `art_*`/`atlas_*` с balanced bagging лучше гибрида с CNN по F1
«предметов»: 0.523 → 0.583. Парно по 30 сидам Δmacro +0.015 (29/30), ΔBA +0.009
(30/30), ΔqF1 +0.009 (30/30), ΔqAUC +0.004 (28/30).

## v5: `spine_positioning` и `spine_axis` → `fused_bag`

У самых редких критериев balanced bagging устойчивее LogReg:

| критерий | позитивов | LogReg (`fused`) | `fused_bag` |
|---|---|---|---|
| `spine_positioning` | 6/99 | укладка 0.477 | **0.486** |
| `spine_axis` | 10/99 | ось 0.403 | **0.416** |

Парно по 30 сидам (v5 против v4): macro Δ+0.005 (26/30, p<0.001),
qc_BA Δ+0.005 (27/30), qc_mF1 Δ+0.006 (28/30), qc_AUC Δ0.000 (p=0.339, не
ухудшено).

`femur_positioning` (36/150) не дисбалансирован — bagging не меняет результат,
оставлен `fused`. Отвергнуты: `geo_bag`/`fused_lda` для `femur_positioning`
(укладка 0.450/0.183), `fused_lda` для `spine_axis` (ось 0.232).

## Итог (принятая конфигурация v5)

```python
CRITERION_SOURCES = {
    "spine_positioning": "fused_bag",   # было fused
    "spine_axis": "fused_bag",          # было fused
    "spine_artifacts": "geo_bag",       # было fused
    "femur_positioning": "fused",
    "femur_roi": "fused_bag",           # было cnn
}
```

**Seed-averaged (30 сидов), base → v5:**

| метрика | base (`d256907`) | v5 | Δ (парно) |
|---|---|---|---|
| organizer macro-F1 | 0.352 | **0.511** | **+0.159** (p<0.001) |
| укладка | 0.477 | 0.486 | +0.009 |
| ось | 0.403 | 0.416 | +0.013 |
| предметы | 0.523 | 0.583 | +0.060 |
| ROI | 0.005 | 0.559 | +0.554 |
| quality_class BA | 0.663 | **0.710** | +0.047 |
| quality_class macro-F1 | 0.649 | **0.688** | +0.039 |
| quality_class ROC-AUC | 0.717 | **0.735** | +0.018 |

**Исторический `fold_id` (`python -m src.evaluate --stacked`), base → v5:**

| метрика | base | v5 | Δ |
|---|---|---|---|
| organizer macro-F1 | 0.439 | **0.468** | +0.029 |
| укладка | 0.476 | **0.494** | +0.018 |
| ось | 0.438 | **0.452** | +0.014 |
| предметы | 0.556 | **0.611** | +0.056 |
| ROI | 0.286 | **0.316** | +0.030 |
| quality_class BA | 0.696 | **0.720** | +0.024 |
| quality_class macro-F1 | 0.675 | **0.694** | +0.019 |
| quality_class ROC-AUC | 0.714 | **0.731** | +0.017 |
| quality head ROC-AUC | 0.586 | 0.586 | 0 |

Ни одна метрика не ухудшилась ни на одном протоколе (qc ROC-AUC v5 против v4 —
в пределах шума: 0.731 против 0.732).

## Отвергнуто (не ухудшать нельзя)

* `spine_positioning=geo_lda` — регрессия по всем метрикам (0/30).
* `spine_axis=geo_bag` — QC-метрики растут (BA 0.718, qF1 0.698, qAUC 0.745), но
  метка «ось» −0.004; в боевую конфигурацию не взято.
* Режимы порога для spine-критериев: `blend`/`f1` хуже `prior` (ось 0.324/0.284
  против 0.403; укладка 0.447/0.378 против 0.477).

## Как воспроизвести

```bash
python scripts/exp_roi_source_real.py --protocol seeds --seeds 30 \
    --map "spine_positioning=fused,spine_axis=fused,spine_artifacts=fused,femur_roi=cnn" \
    --map "spine_positioning=fused_bag,spine_axis=fused_bag,spine_artifacts=geo_bag,femur_roi=fused_bag"
python -m src.stack && python -m src.calibrate && python -m src.evaluate --stacked
```
