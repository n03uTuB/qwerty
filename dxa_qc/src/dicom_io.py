# -*- coding: utf-8 -*-
"""Чтение DICOM, дедупликация изображений, де-идентификация и предобработка.

Модуль соответствует пункту «dicom_parser» из ТЗ. Публичный API полностью
обратно совместим с прежней версией (её используют dataset.py, inference.py,
api.py, visualize.py), добавлены лишь новые функции.

Ключевые наблюдения по данным организатора (НД_для_обучения):
  * исследование -> папка с UID, внутри одна/несколько серий, внутри серии
    подпапки DXA/CR DXA с файлами CR00000N.dcm (без расширения);
  * изображения 8-битные MONOCHROME2, ширина ~300 (поясничный отдел) или
    ~280/248 (проксимальный отдел бедра);
  * одно и то же изображение может быть продублировано -> дедуплицируем по
    содержимому пикселей;
  * сторона бедра однозначно определяется наклоном оси кости:
    положительный -> левое бедро, отрицательный -> правое бедро
    (подтверждено на исследованиях с известной разметкой левого бедра);
  * реальный размер пикселя лежит в теге (0040,0303) Exposed Area, а не в
    PixelSpacing (он пуст) — см. pixel_spacing.

Предобработка (v2) приведена к единому конвейеру для обучения и инференса:

    raw pixel_array
      -> modality LUT (Rescale Slope/Intercept)
      -> VOI LUT / WindowCenter-Width (если есть)
      -> робастная перцентильная нормировка (0.5/99.5)
      -> инверсия для MONOCHROME1
      -> [опционально] CLAHE по костным тканям
      -> квадрат-паддинг -> ресайз -> float32 [0,1]

Де-идентификация: `deidentify()` удаляет ПДн по профилю DICOM PS3.15 (Basic
Application Level Confidentiality Profile), `extract_uids()` извлекает
study/series/image UID (с опциональным хешированием), `hash_uid()` — утилита
псевдонимизации.
"""
from __future__ import annotations

import hashlib
import os
import warnings
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

from . import config as C

warnings.filterwarnings("ignore")

Array = np.ndarray


# --------------------------------------------------------------------------- #
# Поиск и чтение
# --------------------------------------------------------------------------- #
def find_dicom_files(study_dir: str) -> List[str]:
    """Рекурсивно собрать все файлы исследования (DICOM без расширения)."""
    out: List[str] = []
    for root, _dirs, files in os.walk(study_dir):
        for name in files:
            if name.lower().endswith((".xlsx", ".csv", ".json", ".png", ".nii")):
                continue
            out.append(os.path.join(root, name))
    return sorted(out)


def iter_studies(root: str) -> Iterator[Tuple[str, str]]:
    """Перебрать исследования (study_uid, study_path) в корневом каталоге."""
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            yield name, path


def _squeeze_frames(arr: Array) -> Array:
    """Привести мультикадровый массив к одному кадру (DXA всегда одиночный)."""
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[0] if arr.shape[0] >= 1 else arr
    elif arr.ndim > 3:
        arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])[0]
    return arr


def read_dicom(path: str) -> Tuple[object, Optional[Array]]:
    """Прочитать DICOM-файл, вернуть (dataset, raw_pixel_array).

    Возвращает ИСХОДНЫЙ массив (без модального/оконного преобразования) —
    так же, как раньше, чтобы дедупликация и определение области не менялись.
    Для получения готового к показу изображения используйте to_display_float().
    """
    ds = pydicom.dcmread(path, force=True)
    try:
        arr = ds.pixel_array
        arr = _squeeze_frames(arr)
    except Exception:
        arr = None
    return ds, arr


def read_dicom_display(path: str) -> Tuple[object, Optional[Array]]:
    """Прочитать DICOM и вернуть (dataset, float32 [0,1] после modality/VOI LUT)."""
    ds, arr = read_dicom(path)
    if arr is None:
        return ds, None
    return ds, to_display_float(ds, arr)


def pixel_spacing(ds, arr: Array) -> Tuple[float, float]:
    """Размер пикселя (мм по X, мм по Y).

    Сначала пробуем стандартные теги (PixelSpacing / ImagerPixelSpacing /
    NominalScannedPixelSpacing / PixelAspectRatio). В наборе организатора они
    пусты, зато есть (0040,0303) Exposed Area — физический размер снятой области
    в мм. Деление на размер кадра даёт реальный масштаб каждого снимка, поэтому
    углы и сантиметры считаются по факту, а не по предположению.
    """
    # 1) стандартные теги
    for tag in ("PixelSpacing", "ImagerPixelSpacing", "NominalScannedPixelSpacing"):
        try:
            v = getattr(ds, tag, None)
            if v is not None and len(v) >= 2:
                sy, sx = float(v[0]), float(v[1])   # DICOM: [row spacing, col spacing]
                if _spacing_ok(sx) and _spacing_ok(sy):
                    return sx, sy
        except Exception:
            pass
    # 2) Exposed Area (специфика набора организатора)
    try:
        area = ds.get(C.EXPOSED_AREA_TAG)
        if area is None or arr is None or len(area.value) < 2:
            return C.PIXEL_SPACING_MM
        width_mm, height_mm = float(area.value[0]), float(area.value[1])
        rows, cols = arr.shape[:2]
        sx = width_mm / cols if cols and width_mm > 0 else C.PIXEL_SPACING_MM[0]
        sy = height_mm / rows if rows and height_mm > 0 else C.PIXEL_SPACING_MM[1]
    except Exception:
        return C.PIXEL_SPACING_MM
    if not (_spacing_ok(sx) and _spacing_ok(sy)):
        return C.PIXEL_SPACING_MM
    return sx, sy


def _spacing_ok(v: float) -> bool:
    lo, hi = C.PIXEL_SPACING_LIMITS
    return lo <= v <= hi


def dedupe_unique_images(files: List[str]) -> List[Tuple[str, object, Array]]:
    """Уникальные изображения исследования (по содержимому пикселей).

    Возвращает список (representative_path, dataset, array) в порядке
    возрастания InstanceNumber. Число копий каждого изображения доступно
    отдельно через study_copies() (признак «переснятых» исследований).

    Важно: в наборе организатора одно и то же изображение выгружено несколькими
    файлами, и тег Exposed Area у них расходится (например, [170,160] против
    мусорной константы [520,478] -> масштаб завышается втрое, а с ним и все
    сантиметры/площади). Поэтому представителем группы берём файл, чей масштаб
    ближе всего к МЕДИАННОМУ масштабу исследования: настоящие значения лежат
    кучно (~0.6 мм), а выбросы — нет.
    """
    groups: Dict[Tuple[tuple, bytes], List[Tuple[str, object, Array]]] = {}
    for path in files:
        ds, arr = read_dicom(path)
        if arr is None:
            continue
        arr = np.asarray(arr)
        key = (arr.shape, hashlib.md5(np.ascontiguousarray(arr).tobytes()).digest())
        groups.setdefault(key, []).append((path, ds, arr))

    # медианный масштаб исследования — устойчивая оценка «настоящего» размера пикселя
    spacings: List[float] = []
    for cands in groups.values():
        for _p, _ds, _arr in cands:
            try:
                spacings.append(float(pixel_spacing(_ds, _arr)[0]))
            except Exception:
                pass
    median_sx = float(np.median(spacings)) if spacings else C.PIXEL_SPACING_MM[0]

    items: List[Tuple[str, object, Array]] = []
    for cands in groups.values():
        def _dist(t: Tuple[str, object, Array]) -> float:
            try:
                return abs(float(pixel_spacing(t[1], t[2])[0]) - median_sx)
            except Exception:
                return float("inf")
        items.append(min(cands, key=_dist))

    def _inst(t):
        try:
            return int(t[1].InstanceNumber)
        except Exception:
            return 0

    items.sort(key=_inst)
    return items


def study_copies(files: List[str]) -> Dict[str, int]:
    """Сколько раз каждое уникальное изображение встречается в исследовании.

    Ключ — хеш содержимого пикселей; значение — число копий. Используется
    манифестом как дополнительный признак (см. docs/findings.md, находка 3).
    """
    counts: Dict[str, int] = {}
    for path in files:
        _ds, arr = read_dicom(path)
        if arr is None:
            continue
        key = hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()
        counts[key] = counts.get(key, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# Де-идентификация и UID
# --------------------------------------------------------------------------- #
# ПДн-теги по DICOM PS3.15 (Basic Application Level Confidentiality Profile).
PHI_KEYWORDS: Tuple[str, ...] = (
    "PatientName", "PatientID", "IssuerOfPatientID", "PatientBirthDate",
    "PatientBirthTime", "PatientSex", "PatientAge", "PatientSize",
    "PatientWeight", "PatientAddress", "PatientTelephoneNumbers",
    "PatientMotherBirthName", "OtherPatientIDs", "OtherPatientNames",
    "OtherPatientIDsSequence", "PatientBirthName", "PatientComments",
    "MilitaryRank", "EthnicGroup", "Occupation",
    "ReferringPhysicianName", "ReferringPhysicianAddress",
    "ReferringPhysicianTelephoneNumbers", "PhysiciansOfRecord",
    "PerformingPhysicianName", "NameOfPhysiciansReadingStudy",
    "OperatorsName", "RequestingPhysician", "InstitutionName",
    "InstitutionAddress", "InstitutionalDepartmentName", "StationName",
    "AccessionNumber", "StudyID", "DeviceSerialNumber", "ProtocolName",
    "ImageComments", "StudyDescription", "SeriesDescription",
    "RequestAttributesSequence", "PhysiciansOfRecordIdentificationSequence",
)

# Теги, которые НЕ являются ПДн и нужны для связывания/обработки — оставляем.
DEID_WHITELIST: Tuple[str, ...] = (
    "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID",
    "SOPClassUID", "Modality", "InstanceNumber", "SeriesNumber",
    "PhotometricInterpretation", "Rows", "Columns", "BitsAllocated",
    "BitsStored", "PixelRepresentation", "RescaleSlope", "RescaleIntercept",
    "WindowCenter", "WindowWidth", "VOILUTSequence", "BodyPartExamined",
    "ExposedArea", "PixelData",
)


def deidentify(ds, inplace: bool = True):
    """Удалить ПДн из датасета DICOM (профиль PS3.15, базовый уровень).

    UID (Study/Series/SOP) сохраняются как псевдонимизированные идентификаторы:
    без них невозможно связать строки отчёта между собой. Для дополнительной
    защиты используйте hash_uid()/extract_uids(hash_uids=True).
    """
    target = ds if inplace else ds.copy()
    for kw in PHI_KEYWORDS:
        try:
            if kw in target:
                del target[kw]
        except Exception:
            continue
    return target


def hash_uid(uid: str, salt: str = C.UID_SALT) -> str:
    """Детерминированный псевдоним UID (необратимый, с солью)."""
    if uid is None:
        return ""
    return hashlib.sha256((salt + "|" + str(uid)).encode("utf-8")).hexdigest()[:32]


def extract_uids(ds, hash_uids: bool = False, salt: str = C.UID_SALT) -> Dict[str, str]:
    """Извлечь study/series/image UID без утечки ПДн.

    По умолчанию возвращает исходные UID (как ожидает организатор). При
    hash_uids=True значения заменяются детерминированными псевдонимами — это
    исключает возможность повторной идентификации пациента по UID в отчёте.
    """
    study = str(getattr(ds, "StudyInstanceUID", "") or "")
    series = str(getattr(ds, "SeriesInstanceUID", "") or "")
    image = str(getattr(ds, "SOPInstanceUID", "") or "")
    if hash_uids:
        study, series, image = hash_uid(study, salt), hash_uid(series, salt), hash_uid(image, salt)
    return {"study_uid": study, "series_uid": series, "image_uid": image}


# --------------------------------------------------------------------------- #
# Признаки и определение области
# --------------------------------------------------------------------------- #
def bone_slope(arr: Array) -> float:
    """Наклон главной оси яркой (костной) структуры.

    > 0 -> диагональ «верх-лево / низ-право» (левое бедро),
    < 0 -> зеркальная диагональ (правое бедро).
    """
    f = arr.astype(np.float32)
    if f.max() <= f.min():
        return 0.0
    thr = np.percentile(f, 92)
    ys, xs = np.where(f >= thr)
    if len(xs) < 10:
        return 0.0
    h, w = arr.shape
    x = xs / float(w)
    y = ys / float(h)
    xm, ym = x.mean(), y.mean()
    vx, vy = x - xm, y - ym
    cov = np.array(
        [[np.mean(vx * vx), np.mean(vx * vy)], [np.mean(vx * vy), np.mean(vy * vy)]],
        dtype=np.float64,
    )
    vals, vecs = np.linalg.eigh(cov)
    main = vecs[:, int(np.argmax(vals))]
    if main[1] < 0:  # ориентируем вниз (положительное y — вниз)
        main = -main
    return float(main[0] / (main[1] + 1e-6))


def detect_region(arr: Array) -> str:
    """Определить анатомическую область по геометрии изображения (ширина кадра)."""
    cols = arr.shape[1]
    if cols >= C.SPINE_MIN_COLUMNS:
        return C.REGION_SPINE
    return C.REGION_FEMUR_LEFT if bone_slope(arr) >= 0 else C.REGION_FEMUR_RIGHT


def detect_region_study(items: Sequence[Tuple[str, object, Array]]) -> List[str]:
    """Определить области для всех изображений исследования.

    Комбинирует два сигнала: порядок съёмки (InstanceNumber) и ширину кадра.
    Находка по данным: позвоночник почти всегда снят первым (InstanceNumber=1
    в 91 из 95 исследований), что устойчивее правила «ширина >= 295» при смене
    аппарата. Используется только при C.REGION_BY_INSTANCE_ORDER=True.
    """
    regions = [detect_region(arr) for _p, _ds, arr in items]
    if not C.REGION_BY_INSTANCE_ORDER or not items:
        return regions
    # если среди изображений ровно одно «широкое» и оно снято первым — считаем его
    # позвоночником, а бедро(а) переназначаем по наклону оси.
    try:
        inst = [int(getattr(ds, "InstanceNumber", 0) or 0) for _p, ds, _a in items]
    except Exception:
        return regions
    if len(items) < 2:
        return regions
    first = int(np.argmin(inst))
    if regions[first] == C.REGION_SPINE:
        return regions
    # порядок подсказывает позвоночник, но ширина говорит «бедро»: доверяем порядку
    regions[first] = C.REGION_SPINE
    for i, (_p, _ds, arr) in enumerate(items):
        if i != first and regions[i] == C.REGION_SPINE:
            regions[i] = C.REGION_FEMUR_LEFT if bone_slope(arr) >= 0 else C.REGION_FEMUR_RIGHT
    return regions


# --------------------------------------------------------------------------- #
# Предобработка под нейросеть
# --------------------------------------------------------------------------- #
def modality_lut(ds, arr: Array) -> Array:
    """Применить Rescale Slope/Intercept (модальное преобразование DICOM)."""
    f = np.asarray(arr).astype(np.float32)
    try:
        from pydicom.pixels import apply_modality_lut

        return apply_modality_lut(f, ds).astype(np.float32)
    except Exception:
        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
        return f * slope + intercept if (slope != 1.0 or intercept != 0.0) else f


def voi_lut(ds, arr: Array) -> Array:
    """Применить VOI LUT / окно (WindowCenter/WindowWidth), если оно задано.

    Для данных организатора окна нет — возвращается вход без изменений.
    """
    f = np.asarray(arr).astype(np.float32)
    has_voi = (getattr(ds, "VOILUTSequence", None) is not None
               or getattr(ds, "WindowCenter", None) is not None)
    if not has_voi:
        return f
    try:
        from pydicom.pixels import apply_voi_lut

        return apply_voi_lut(f, ds, index=0).astype(np.float32)
    except Exception:
        return f


def to_display_float(ds, arr: Array) -> Array:
    """Единый конвейер: modality LUT -> VOI LUT -> нормировка -> MONOCHROME1.

    Возвращает float32 [0,1] той же формы, что и вход.
    """
    f = modality_lut(ds, arr)
    f = voi_lut(ds, f)
    f = f.astype(np.float32)
    lo, hi = np.percentile(f, 0.5), np.percentile(f, 99.5)
    if hi - lo < 1e-6:
        hi = lo + 1.0
    f = np.clip((f - lo) / (hi - lo), 0.0, 1.0)
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        f = 1.0 - f
    return f.astype(np.float32)


def to_display_uint8(ds, arr: Array) -> Array:
    """to_display_float + перевод в uint8 (для кэша PNG)."""
    return (to_display_float(ds, arr) * 255.0).round().astype(np.uint8)


def apply_clahe(img01: Array, clip_limit: float = C.CLAHE_CLIP,
                tile: int = C.CLAHE_TILE) -> Array:
    """Локальное выравнивание гистограммы (CLAHE) для костных тканей.

    Повышает локальный контраст трабекулярной структуры и контуров кости —
    помогает и человеку, и сети. Применяется ОДИНАКОВО при обучении и инференсе.
    """
    try:
        import cv2

        u8 = (np.clip(img01, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=float(clip_limit),
                                tileGridSize=(int(tile), int(tile)))
        return (clahe.apply(u8).astype(np.float32) / 255.0)
    except Exception:
        return np.clip(img01, 0.0, 1.0).astype(np.float32)


def normalize_uint8(arr: Array) -> Array:
    """Робастная нормировка яркости в uint8 (для кэша и визуализации)."""
    f = arr.astype(np.float32)
    lo, hi = np.percentile(f, 0.5), np.percentile(f, 99.5)
    if hi - lo < 1e-6:
        hi = lo + 1.0
    f = np.clip((f - lo) / (hi - lo), 0.0, 1.0)
    return (f * 255.0).round().astype(np.uint8)


def pad_to_square(arr: Array, fill: int = 0) -> Array:
    """Дополнить изображение до квадрата по центру (сохраняя пропорции)."""
    h, w = arr.shape
    s = max(h, w)
    out = np.full((s, s), fill, dtype=arr.dtype)
    y0 = (s - h) // 2
    x0 = (s - w) // 2
    out[y0:y0 + h, x0:x0 + w] = arr
    return out


def preprocess(arr: Array, size: int = C.IMAGE_SIZE,
               use_clahe: Optional[bool] = None,
               is_display: bool = False,
               ds=None) -> Array:
    """Нормировать, дополнить до квадрата и масштабировать.

    Параметры
    ---------
    arr : сырой пиксельный массив ИЛИ уже нормализованное изображение.
    use_clahe : применить CLAHE (по умолчанию — C.USE_CLAHE).
    is_display : arr уже приведён к диапазону [0,1] или uint8 (кэш PNG) —
        повторная перцентильная нормировка не нужна.
    ds : датасет DICOM; если передан, к arr применяется modality/VOI LUT.

    Возвращает float32 [0,1] формы (size, size).
    """
    import cv2

    if use_clahe is None:
        use_clahe = C.USE_CLAHE

    if ds is not None:
        img = to_display_float(ds, arr)
    elif is_display:
        img = np.asarray(arr).astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        img = np.clip(img, 0.0, 1.0)
    else:
        img = normalize_uint8(arr).astype(np.float32) / 255.0

    if use_clahe:
        img = apply_clahe(img, C.CLAHE_CLIP, C.CLAHE_TILE)

    u8 = (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    u8 = pad_to_square(u8, fill=0)
    interp = cv2.INTER_AREA if u8.shape[0] > size else cv2.INTER_LINEAR
    u8 = cv2.resize(u8, (size, size), interpolation=interp)
    return u8.astype(np.float32) / 255.0
