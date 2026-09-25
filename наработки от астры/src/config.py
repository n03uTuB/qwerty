# -*- coding: utf-8 -*-
"""Единая конфигурация гибридного сервиса DXA-QC.

Единственный источник правды для путей, таксономии, гиперпараметров и —
главное — для выбора источников/признаков по каждому критерию ТЗ.

Откуда взяты значения:
  * базовая таксономия, формат вывода и масштаб пикселя — из решения команды
    (``dxa_qc``), проверены на наборе организатора (см. ``docs/findings.md``);
  * ``CRITERION_SOURCES`` / ``CRITERION_FEATURES_OVERRIDE`` — конфигурация v3
    (``opus_solution``): +0.056 macro-F1, подтверждено парно на 20 разбиениях;
  * ``PREPROCESS_MODE`` / ``USE_BONE_WINDOW`` / ``USE_DENOISE`` и режим порога
    ``blend`` — из ``fable_solution``;
  * ``infer_backbone_from_weights`` (см. :mod:`src.models.backbones`) — тоже
    от ``fable_solution``: устраняет рассинхрон весов и бэкбона.
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# Пути
# --------------------------------------------------------------------------- #
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Набор организатора «НД_для_обучения» может лежать рядом с корнем проекта либо
# на уровень выше (когда решение вынесено в подпапку, напр. «наработки от астры»).
_DATASET_CANDIDATES = [
    os.environ.get("DXA_DATASET"),
    os.path.join(os.path.dirname(PROJECT_ROOT), "НД_для_обучения"),
    os.path.join(os.path.dirname(os.path.dirname(PROJECT_ROOT)), "НД_для_обучения"),
]
DEFAULT_DATASET_ROOT = next(
    (c for c in _DATASET_CANDIDATES if c and os.path.isdir(c)),
    _DATASET_CANDIDATES[1],
)

ARTIFACTS_DIR = os.environ.get("DXA_ARTIFACTS", os.path.join(PROJECT_ROOT, "artifacts"))
CACHE_DIR = os.path.join(ARTIFACTS_DIR, "image_cache")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

MANIFEST_CSV = os.path.join(ARTIFACTS_DIR, "manifest.csv")
FEATURES_CSV = os.path.join(ARTIFACTS_DIR, "features.csv")
BEST_WEIGHTS = os.path.join(ARTIFACTS_DIR, "best_model.pt")
FOLD_WEIGHTS_TMPL = os.path.join(ARTIFACTS_DIR, "fold_{k}.pt")
METRICS_JSON = os.path.join(ARTIFACTS_DIR, "metrics.json")
OOF_QUALITY_NPY = os.path.join(ARTIFACTS_DIR, "oof_quality.npy")
OOF_VIOLATION_NPY = os.path.join(ARTIFACTS_DIR, "oof_violation.npy")
OOF_REGION_NPY = os.path.join(ARTIFACTS_DIR, "oof_region.npy")
STACK_OOF_VIOLATION_NPY = os.path.join(ARTIFACTS_DIR, "stack_oof_violation.npy")
STACKER_PATH = os.path.join(ARTIFACTS_DIR, "stacker.joblib")
STACK_METRICS = os.path.join(ARTIFACTS_DIR, "stack_metrics.json")
THRESHOLDS_JSON = os.path.join(ARTIFACTS_DIR, "thresholds.json")

# --------------------------------------------------------------------------- #
# Таксономия
# --------------------------------------------------------------------------- #
REGION_SPINE = "lumbar_spine"
REGION_FEMUR_LEFT = "proximal_femur_left"
REGION_FEMUR_RIGHT = "proximal_femur_right"
REGIONS = [REGION_SPINE, REGION_FEMUR_LEFT, REGION_FEMUR_RIGHT]
REGION_TO_IDX = {r: i for i, r in enumerate(REGIONS)}

# Мультилейбл-вектор критериев (индексы фиксированы — менять нельзя).
VIOLATIONS = [
    "spine_positioning",   # 0 укладка поясничного отдела
    "spine_axis",          # 1 ось позвоночника (<= 5 град)
    "spine_artifacts",     # 2 посторонние предметы / артефакты / наложения
    "femur_positioning",   # 3 позиционирование / ротация бедра
    "femur_roi",           # 4 корректность области интереса (ROI)
]
N_VIOLATIONS = len(VIOLATIONS)
VIOLATION_IDX = {v: i for i, v in enumerate(VIOLATIONS)}

VIOLATION_HUMAN = {
    "spine_positioning": "нарушение укладки поясничного отдела",
    "spine_axis": "отклонение оси позвоночника (>5 град)",
    "spine_artifacts": "посторонние предметы/артефакты/наложения",
    "femur_positioning": "нарушение позиционирования/ротации бедра",
    "femur_roi": "некорректная область интереса (ROI)",
}

# Какие критерии применимы к какой области (для маскирования лосса и решения).
REGION_CRITERIA = {
    REGION_SPINE: ["spine_positioning", "spine_axis", "spine_artifacts"],
    REGION_FEMUR_LEFT: ["femur_positioning", "femur_roi"],
    REGION_FEMUR_RIGHT: ["femur_positioning", "femur_roi"],
}

# --------------------------------------------------------------------------- #
# Формат вывода (по разъяснениям организатора, ред. V2)
# --------------------------------------------------------------------------- #
REGION_LABEL_RU = {
    REGION_SPINE: "Поясничный отдел позвоночника",
    REGION_FEMUR_LEFT: "Проксимальный отдел бедра",
    REGION_FEMUR_RIGHT: "Проксимальный отдел бедра",
}

# Закрытый список violation_type. «Некорректная укладка» общая для двух областей.
VIOLATION_LABEL_RU = {
    "spine_positioning": "Некорректная укладка",
    "spine_axis": "Не выравнена ось позвоночника",
    "spine_artifacts": "Присутствуют посторонние предметы",
    "femur_positioning": "Некорректная укладка",
    "femur_roi": "Некорректная область интереса",
}
VIOLATION_LABELS_UNIQUE = [
    "Некорректная укладка",
    "Не выравнена ось позвоночника",
    "Присутствуют посторонние предметы",
    "Некорректная область интереса",
]
VIOLATION_SEP = "; "
VIOLATION_EMPTY = ""

# Обязательные колонки отчёта по ТЗ (порядок фиксирован).
REPORT_COLUMNS = [
    "path_to_study", "study_uid", "image_uid", "anatomical_region",
    "quality_class", "violation_type", "processing_status", "time_of_processing",
]
# Дополнительные (не обязательные) колонки — вероятности для калибровки и аудита.
REPORT_EXTRA_COLUMNS = ["quality_prob", "violation_probs"]

# Метки организатора: каждая = OR своих критериев (по ним считается macro-F1).
ORG_LABELS = {
    "укладка": ["spine_positioning", "femur_positioning"],
    "ось": ["spine_axis"],
    "предметы": ["spine_artifacts"],
    "ROI": ["femur_roi"],
}

# --------------------------------------------------------------------------- #
# Масштаб пикселя (находка 1 и 9 из docs/findings.md)
# --------------------------------------------------------------------------- #
# PixelSpacing пуст; физический размер снятой области лежит в (0040,0303)
# Exposed Area. Деление на размер кадра даёт реальный масштаб каждого снимка
# (~0.607 x 0.608 мм). Значение ниже — только запасное.
PIXEL_SPACING_MM = (0.6, 0.6)      # (x, y)
EXPOSED_AREA_TAG = (0x0040, 0x0303)
# Отсев мусорного тега [520,478] (давал 1.7-1.9 мм -> утечка метки, находка 9).
PIXEL_SPACING_LIMITS = (0.2, 1.2)

# Определение области: ширина кадра (позвоночник шире) + наклон оси кости (сторона).
SPINE_MIN_COLUMNS = 295
# Порядок съёмки (InstanceNumber=1 -> позвоночник) устойчивее правила по ширине,
# но включается только после проверки на своих данных.
REGION_BY_INSTANCE_ORDER = False

# --------------------------------------------------------------------------- #
# Модель и обучение
# --------------------------------------------------------------------------- #
IMAGE_SIZE = 224
# Базовый бэкбон. На 252 снимках крупные сети переобучаются (проверено), поэтому
# resnet18 — дефолт; convnext_tiny / swin_tiny_patch4_window7_224 доступны через timm.
BACKBONE = "resnet18"
BACKBONE_ALTERNATIVES = ["resnet18", "resnet34", "efficientnet_b0",
                         "convnext_tiny", "swin_tiny_patch4_window7_224"]
# Ансамбль по бэкбонам (fable_solution): усреднение вероятностей resnet18+resnet34.
ENSEMBLE_BACKBONES = ["resnet18", "resnet34"]
DROPOUT = 0.3

BATCH_SIZE = 24
EPOCHS = 24
LR = 2e-4
WEIGHT_DECAY = 1e-4
N_FOLDS = 5
SEED = 42
NUM_WORKERS = 0            # Windows-friendly; на Linux/H200 поднять до 8-16

# Функция потерь: focal (по умолчанию, для сильного дисбаланса 6-36 позитивов)
# | weighted_bce (воспроизводит базовое решение команды).
QUALITY_LOSS = "focal"
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.5

# Смешанная точность (H200/Hopper) и TTA при оценке/инференсе.
USE_AMP = True
USE_TTA = True
# Безопасный TTA: малые повороты (<5 град — не имитируют нарушение оси) и лёгкий зум.
# Отражения НЕ используем: они меняют анатомическую сторону (лево/право) бедра.
TTA_AFFINES = [(0.0, 1.0), (2.0, 1.0), (-2.0, 1.0), (0.0, 0.95), (0.0, 1.05)]

# --------------------------------------------------------------------------- #
# Предобработка (fable_solution)
# --------------------------------------------------------------------------- #
# Режим входа сети: "bone" (костное окно + denoise+CLAHE — рекомендовано) |
# "display" (как базовое решение: перцентильная нормировка + CLAHE).
PREPROCESS_MODE = "bone"
USE_BONE_WINDOW = True
USE_DENOISE = True         # bilateralFilter перед CLAHE (убирает зерно)
USE_CLAHE = True
CLAHE_CLIP = 2.0
CLAHE_TILE = 8

# --------------------------------------------------------------------------- #
# Валидация и пороги
# --------------------------------------------------------------------------- #
# StratifiedGroupKFold сохраняет баланс областей/классов внутри фолдов.
USE_STRATIFIED_GROUP = True
NESTED_THRESHOLD = True
NESTED_SPLITS = 5

# Как выбирать порог классификации (calibrate.py + оценка):
#   "f1"    — порог, максимизирующий F1 на train-части (дефолт: воспроизводит
#             подтверждённый v3 macro-F1 0.563 на 20 разбиениях, opus_solution);
#   "blend" — 0.5*(prior + f1): устойчив к калибровке, значимо лучше prior
#             (+0.017..+0.021, CI не накрывает 0; fable_solution);
#   "prior" — столько снимков, сколько нарушений ожидается по доле;
#   "nested"— порог по внутренней CV (самый консервативный);
#   "mapped"— режим задаётся ПО КРИТЕРИЮ (см. THRESHOLD_MODE_BY_CRITERION).
THRESHOLD_MODE = "mapped"

# Режим порога по каждому критерию (действует при THRESHOLD_MODE="mapped").
#
# Почему per-criterion. Метрика организатора — macro-F1 по OR-меткам, а F1-порог
# при 6-36 позитивах нестабилен (argmax на train переобучается и на val проваливается).
# Порог по доле ("prior") устойчив и на 3 из 4 меток значимо лучше: честный
# кластер-бутстрэп по исследованиям (B=1000, scripts/diag_threshold_final.py)
# даёт macro-F1 0.344 -> 0.441 (95% ДИ прироста [+0.026, +0.164], P(>0)=0.99)
# и quality_class Balanced Accuracy 0.623 -> 0.704 (P=0.99), ROC-AUC не меняется
# (0.714 — порог на него не влияет).
#
# Исключение — femur_roi (7 позитивов): порог по доле вырождается (топ-7 по скору
# не пересекаются с истиной ни в одном val-фолде), поэтому там оставлен F1-порог.
# Он НЕ ухудшает ROI (F1 0.286 сохраняется) и не создаёт основной прирост
# (прирост дают prior на укладке/оси/предметах).
THRESHOLD_MODE_BY_CRITERION = {
    "spine_positioning": "prior",
    "spine_axis": "prior",
    "spine_artifacts": "prior",
    "femur_positioning": "prior",
    "femur_roi": "f1",
}

# Балансировка классов сэмплером в дополнение к pos_weight лосса.
BALANCED_SAMPLER = False

# --------------------------------------------------------------------------- #
# Источник скора по критерию (КОНФИГУРАЦИЯ v3 — ключевое улучшение)
# --------------------------------------------------------------------------- #
#   "cnn"   — чистая нейросеть;
#   "fused" — гибрид CNN + геометрия (LogReg: вероятность CNN вместе с признаками);
#   "geo"   — чистая геометрия (LogReg только по признакам).
#
# v3 (opus_solution, честно на 20 разбиениях StratifiedGroupKFold по study_uid):
#   * spine_axis: "fused" + spine_midline_residual -> F1 «ось» 0.423 -> 0.620
#     (20/20 разбиений). Остаточная кривизна средней линии отличает «наклон от
#     прямой оси» от «физиологического изгиба» — ранжирование становится
#     разделяющим (репозиторий сам отмечал «AUC оси 0.82, а F1 0.42»);
#   * spine_artifacts: "fused" вместо "cnn" -> AUC критерия 0.557 -> 0.700;
#   * femur_positioning: "geo" (чистая геометрия) — парный тест +0.013 (20/20),
#     строгая вложенная проверка подтвердила +0.054 по метке «укладка»;
#   * итог: macro-F1 0.507 -> 0.563 (+0.056, 19/20).
# spine_positioning: s novymi atlas_/art_ priznakami "fused" chestno luchshe
# cnn: AUC 0.758 -> 0.840, macro-F1 0.286 -> 0.513 (10x5 GroupKFold,
# scripts/check_positioning_source.py).
CRITERION_SOURCES = {
    "spine_positioning": "fused",
    "spine_axis": "fused",
    "spine_artifacts": "fused",
    "femur_positioning": "fused",
    "femur_roi": "cnn",
}

# Переопределение геометрических наборов для гибридного стекера (stack.py).
# Пусто -> наборы по умолчанию из stack.CRITERION_FEATURES.
#
# v3 (подтверждено): spine_axis = угол средней линии + остаточная кривизна.
# Новые семейства (по запросу — «выделить кость, остальное = предметы»):
#   * art_*   — яркие объекты/резкие края ВНЕ костной маски, тёмное внутри тела;
#   * atlas_* — остаток относительно костного шаблона (медиана зарегистрированных
#     снимков), align_* — наклон/масштаб, потребовавшиеся при регистрации.
CRITERION_FEATURES_OVERRIDE = {
    "spine_axis": ["spine_midline_angle", "spine_midline_residual"],
    # predmety: rez kostej vne kosti + sovpadenije s kostnym shablonom.
    # Na realnyh dannyh (10x5 GroupKFold): AUC 0.706->0.847, F1 0.423->0.574.
    "spine_artifacts": [
        "art_edge_out_p99", "atlas_bone_iou",
        "atlas_resid_out_p99", "art_bright_out_edge",
    ],
    # ukladka: podvzdoshnyj signal + pikovaya jarkost vne kosti + vybrosy k
    # shablonu + ekscentrichnost kosti.
    # Na realnyh dannyh: AUC 0.785->0.832, F1 0.218->0.340.
    "spine_positioning": [
        "spine_iliac_signal", "art_max_out",
        "atlas_outlier_area", "bone_eccentricity",
    ],
    # укладка бедра (ротация/позиционирование): ширина кадра (FOV) + наклон оси
    # кости + доля высоты кадра, занятая костью. На реальных данных
    # (10x5 GroupKFold, scripts/eval_femur_final.py): AUC 0.630 -> 0.676,
    # при этом другие критерии не затрагиваются (признаки бедра локальны).
    "femur_positioning": [
        "femur_width_cm", "femur_axis_angle", "femur_bbox_h_ratio",
    ],
}

# Признак «число копий снимка» связан с нарушением (находка 3), но это метаданные
# выгрузки: на закрытом тесте может не воспроизвестись. Оставляем колонку в
# манифесте для анализа, но в стекер по умолчанию НЕ подаём (аудит утечек, fable).
USE_COPIES_FEATURE = False

# Псевдонимизация UID в отчёте (доп. защита ПДн). По умолчанию выключена: организатор
# ожидает исходные study_uid/image_uid.
HASH_UIDS_IN_OUTPUT = False
UID_SALT = "dxa-qc-hybrid"

# --------------------------------------------------------------------------- #
# Сервис
# --------------------------------------------------------------------------- #
API_HOST = "0.0.0.0"
API_PORT = 8000
# Лимит ТЗ: не более 3 минут на исследование (страховка для батча).
STUDY_TIME_LIMIT_S = 180.0
