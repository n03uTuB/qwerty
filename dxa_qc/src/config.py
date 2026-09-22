# -*- coding: utf-8 -*-
"""Глобальная конфигурация проекта DXA-QC.

Единый источник правды для путей, таксономии нарушений и гиперпараметров.
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# Пути
# --------------------------------------------------------------------------- #
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET_ROOT = os.path.join(
    os.path.dirname(PROJECT_ROOT), "НД_для_обучения"
)

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
CACHE_DIR = os.path.join(ARTIFACTS_DIR, "image_cache")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

MANIFEST_CSV = os.path.join(ARTIFACTS_DIR, "manifest.csv")
BEST_WEIGHTS = os.path.join(ARTIFACTS_DIR, "best_model.pt")
FOLD_WEIGHTS_TMPL = os.path.join(ARTIFACTS_DIR, "fold_{k}.pt")
METRICS_JSON = os.path.join(ARTIFACTS_DIR, "metrics.json")

# --------------------------------------------------------------------------- #
# Таксономия
# --------------------------------------------------------------------------- #
# Анатомические области
REGION_SPINE = "lumbar_spine"
REGION_FEMUR_LEFT = "proximal_femur_left"
REGION_FEMUR_RIGHT = "proximal_femur_right"
REGIONS = [REGION_SPINE, REGION_FEMUR_LEFT, REGION_FEMUR_RIGHT]
REGION_TO_IDX = {r: i for i, r in enumerate(REGIONS)}

# Типы нарушений (единый мультилейбл-вектор, индексы фиксированы)
VIOLATIONS = [
    "spine_positioning",   # 0 корректная укладка поясничного отдела
    "spine_axis",          # 1 правильно выровненная ось позвоночника (<=5 град)
    "spine_artifacts",     # 2 посторонние предметы / артефакты / наложения
    "femur_positioning",   # 3 позиционирование / ротация проксимального отдела бедра
    "femur_roi",           # 4 корректность области интереса (ROI)
]
N_VIOLATIONS = len(VIOLATIONS)

VIOLATION_HUMAN = {
    "spine_positioning": "нарушение укладки поясничного отдела",
    "spine_axis": "отклонение оси позвоночника (>5 град)",
    "spine_artifacts": "посторонние предметы/артефакты/наложения",
    "femur_positioning": "нарушение позиционирования/ротации бедра",
    "femur_roi": "некорректная область интереса (ROI)",
}

# Какие критерии относятся к какой области (для маскирования лосса)
REGION_CRITERIA = {
    REGION_SPINE: ["spine_positioning", "spine_axis", "spine_artifacts"],
    REGION_FEMUR_LEFT: ["femur_positioning", "femur_roi"],
    REGION_FEMUR_RIGHT: ["femur_positioning", "femur_roi"],
}
VIOLATION_IDX = {v: i for i, v in enumerate(VIOLATIONS)}

# --------------------------------------------------------------------------- #
# Формат вывода (по разъяснениям организатора, ред. V2)
# --------------------------------------------------------------------------- #
# Русские названия областей: сторона (лево/право) не указывается.
REGION_LABEL_RU = {
    REGION_SPINE: "Поясничный отдел позвоночника",
    REGION_FEMUR_LEFT: "Проксимальный отдел бедра",
    REGION_FEMUR_RIGHT: "Проксимальный отдел бедра",
}

# Закрытый список значений violation_type (организатор).
# ВНИМАНИЕ: «Некорректная укладка» общая для позвоночника и бедра.
VIOLATION_LABEL_RU = {
    "spine_positioning": "Некорректная укладка",
    "spine_axis": "Не выравнена ось позвоночника",
    "spine_artifacts": "Присутствуют посторонние предметы",
    "femur_positioning": "Некорректная укладка",
    "femur_roi": "Некорректная область интереса",
}
# Уникальные метки, по которым считается Macro-F1 у организатора (4 шт.)
VIOLATION_LABELS_UNIQUE = [
    "Некорректная укладка",
    "Не выравнена ось позвоночника",
    "Присутствуют посторонние предметы",
    "Некорректная область интереса",
]
VIOLATION_SEP = "; "        # разделитель при нескольких нарушениях
VIOLATION_EMPTY = ""        # при отсутствии нарушений поле пустое

# Размер пикселя берётся ИЗ САМОГО СНИМКА: тег (0040,0303) Exposed Area хранит физический
# размер снятой области в мм. Тег присутствует во всех 499 файлах набора; деление на размер
# кадра даёт пиксель 0.607 x 0.608 мм, то есть КВАДРАТНЫЙ (а не 0.6 x 1.05, как считалось).
# Прежнее значение занижало все углы в 1.75 раза: наклон оси 5° измерялся как 2.9°.
# Значение ниже используется только как запасное, если тега нет.
PIXEL_SPACING_MM = (0.6, 0.6)   # (x, y)
EXPOSED_AREA_TAG = (0x0040, 0x0303)
PIXEL_SPACING_LIMITS = (0.2, 2.0)   # отсев мусорных значений тега

# --------------------------------------------------------------------------- #
# Гиперпараметры модели/обучения
# --------------------------------------------------------------------------- #
IMAGE_SIZE = 224          # вход квадратный (совпадает с обучением/инференсом)
BACKBONE = "resnet18"
BATCH_SIZE = 24
EPOCHS = 24
LR = 2e-4
WEIGHT_DECAY = 1e-4
N_FOLDS = 5
SEED = 42
NUM_WORKERS = 0           # Windows-friendly

# Геометрические пороги определения области
SPINE_MIN_COLUMNS = 295   # ширина >= порога -> поясничный отдел (300 px)
