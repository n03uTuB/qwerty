# -*- coding: utf-8 -*-
"""Чтение DICOM, дедупликация, де-идентификация и определение области.

Порт базового модуля команды (``dxa_qc/src/dicom_io.py``) с одним изменением:
весь пиксельный конвейер (modality/VOI LUT, нормировка, костное окно, denoise,
CLAHE, pad-to-square, resize) вынесен в :mod:`src.preprocess` и вызывается
через тонкие обёртки :func:`to_display_float`, :func:`to_display_uint8`,
:func:`preprocess`. Так обучение и инференс гарантированно используют один и
тот же вход сети (нет рассинхрона train/inference).

Ключевые наблюдения по данным организатора:
  * исследование -> папка с UID, внутри серии, внутри подпапки DXA/CR DXA;
  * изображения 8-битные MONOCHROME2, ширина ~300 (позвоночник) или ~280/248
    (проксимальный отдел бедра);
  * одно изображение может быть продублировано -> дедуп по содержимому;
  * сторона бедра определяется наклоном оси кости (>=0 -> левое);
  * реальный размер пикселя — в теге (0040,0303) Exposed Area (PixelSpacing пуст).
"""
from __future__ import annotations

import hashlib
import os
import warnings
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pydicom

from . import config as C
from . import preprocess as _pp

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
    """Прочитать DICOM -> (dataset, raw_pixel_array без LUT)."""
    ds = pydicom.dcmread(path, force=True)
    try:
        arr = _squeeze_frames(ds.pixel_array)
    except Exception:
        arr = None
    return ds, arr


def read_dicom_display(path: str) -> Tuple[object, Optional[Array]]:
    """Прочитать DICOM -> (dataset, float32 [0,1] после modality/VOI LUT)."""
    ds, arr = read_dicom(path)
    if arr is None:
        return ds, None
    return ds, _pp.to_display_float(ds, arr)


def pixel_spacing(ds, arr: Array) -> Tuple[float, float]:
    """Размер пикселя (мм по X, мм по Y).

    Сначала стандартные теги; в наборе организатора они пусты, зато есть
    (0040,0303) Exposed Area — физический размер снятой области в мм.
    """
    for tag in ("PixelSpacing", "ImagerPixelSpacing", "NominalScannedPixelSpacing"):
        try:
            v = getattr(ds, tag, None)
            if v is not None and len(v) >= 2:
                sy, sx = float(v[0]), float(v[1])   # DICOM: [row, col]
                if _spacing_ok(sx) and _spacing_ok(sy):
                    return sx, sy
        except Exception:
            pass
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

    Представителем группы берём файл, чей масштаб ближе к медианному масштабу
    исследования: настоящие значения лежат кучно (~0.6 мм), а мусорный тег
    [520,478] даёт выброс (утечка метки, находка 9).
    """
    groups: Dict[Tuple[tuple, bytes], List[Tuple[str, object, Array]]] = {}
    for path in files:
        ds, arr = read_dicom(path)
        if arr is None:
            continue
        arr = np.asarray(arr)
        key = (arr.shape, hashlib.md5(np.ascontiguousarray(arr).tobytes()).digest())
        groups.setdefault(key, []).append((path, ds, arr))

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
    """Сколько раз каждое уникальное изображение встречается в исследовании."""
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


def deidentify(ds, inplace: bool = True):
    """Удалить ПДн (профиль DICOM PS3.15, базовый уровень).

    UID (Study/Series/SOP) сохраняются: без них невозможно связать строки
    отчёта. Для доп. защиты используйте hash_uid()/extract_uids(hash_uids=True).
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


def extract_uids(ds, hash_uids: bool = False,
                 salt: str = C.UID_SALT) -> Dict[str, str]:
    """Извлечь study/series/image UID (с опциональной псевдонимизацией)."""
    study = str(getattr(ds, "StudyInstanceUID", "") or "")
    series = str(getattr(ds, "SeriesInstanceUID", "") or "")
    image = str(getattr(ds, "SOPInstanceUID", "") or "")
    if hash_uids:
        study = hash_uid(study, salt)
        series = hash_uid(series, salt)
        image = hash_uid(image, salt)
    return {"study_uid": study, "series_uid": series, "image_uid": image}


# --------------------------------------------------------------------------- #
# Определение области
# --------------------------------------------------------------------------- #
def bone_slope(arr: Array) -> float:
    """Наклон главной оси яркой (костной) структуры.

    > 0 -> диагональ «верх-лево / низ-право» (левое бедро), < 0 -> правое.
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
    if main[1] < 0:
        main = -main
    return float(main[0] / (main[1] + 1e-6))


def detect_region(arr: Array) -> str:
    """Определить анатомическую область по геометрии (ширина кадра + наклон)."""
    cols = arr.shape[1]
    if cols >= C.SPINE_MIN_COLUMNS:
        return C.REGION_SPINE
    return C.REGION_FEMUR_LEFT if bone_slope(arr) >= 0 else C.REGION_FEMUR_RIGHT


def detect_region_study(items: Sequence[Tuple[str, object, Array]]) -> List[str]:
    """Области для всех изображений исследования (ширина + опц. порядок съёмки)."""
    regions = [detect_region(arr) for _p, _ds, arr in items]
    if not C.REGION_BY_INSTANCE_ORDER or not items:
        return regions
    try:
        inst = [int(getattr(ds, "InstanceNumber", 0) or 0) for _p, ds, _a in items]
    except Exception:
        return regions
    if len(items) < 2:
        return regions
    first = int(np.argmin(inst))
    if regions[first] == C.REGION_SPINE:
        return regions
    regions[first] = C.REGION_SPINE
    for i, (_p, _ds, arr) in enumerate(items):
        if i != first and regions[i] == C.REGION_SPINE:
            regions[i] = (C.REGION_FEMUR_LEFT if bone_slope(arr) >= 0
                          else C.REGION_FEMUR_RIGHT)
    return regions


# --------------------------------------------------------------------------- #
# Обёртки над src.preprocess (единый пиксельный конвейер)
# --------------------------------------------------------------------------- #
def to_display_float(ds, arr: Array) -> Array:
    """modality LUT -> VOI LUT -> нормировка -> MONOCHROME1 (float32 [0,1])."""
    return _pp.to_display_float(ds, arr)


def to_display_uint8(ds, arr: Array) -> Array:
    """to_display_float + перевод в uint8 (для PNG-кэша)."""
    return _pp.to_display_uint8(ds, arr)


def preprocess(arr: Array, size: int = C.IMAGE_SIZE,
               use_clahe: Optional[bool] = None, is_display: bool = False,
               ds=None, mode: Optional[str] = None) -> Array:
    """Сырой массив ИЛИ дисплейное изображение -> вход сети (size, size) float32.

    Тонкая обёртка над :func:`src.preprocess.preprocess_for_net`: если передан
    ``ds`` — сначала применяется modality/VOI LUT; если ``is_display`` — вход
    уже в [0,1]/uint8 (например, PNG-кэш).
    """
    if ds is not None:
        display = _pp.to_display_float(ds, arr)
    elif is_display:
        display = np.asarray(arr, dtype=np.float32)
        if display.max() > 1.0:
            display = display / 255.0
    else:
        display = _pp._percentile_norm(arr)  # noqa: SLF001 (внутренняя утилита)
    x = _pp.preprocess_for_net(display, mode=mode, size=size)
    return x[0]