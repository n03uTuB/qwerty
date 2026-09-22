# -*- coding: utf-8 -*-
"""Чтение DICOM, дедупликация изображений и определение анатомической области.

Ключевые наблюдения по данным организатора (НД_для_обучения):
  * исследование -> папка с UID, внутри одна/несколько серий, внутри серии
    подпапки DXA/CR DXA с файлами CR00000N.dcm (без расширения);
  * изображения 8-битные MONOCHROME2, ширина ~300 (поясничный отдел) или
    ~280/248 (проксимальный отдел бедра);
  * одно и то же изображение может быть продублировано -> дедуплицируем по
    содержимому пикселей;
  * сторона бедра однозначно определяется наклоном оси кости:
    положительный -> левое бедро, отрицательный -> правое бедро
    (подтверждено на исследованиях с известной разметкой левого бедра).
"""
from __future__ import annotations

import os
import warnings
from typing import Iterator, List, Tuple

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
            out.append(os.path.join(root, name))
    return sorted(out)


def iter_studies(root: str) -> Iterator[Tuple[str, str]]:
    """Перебрать исследования (study_uid, study_path) в корневом каталоге."""
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            yield name, path


def read_dicom(path: str):
    """Прочитать DICOM-файл, вернуть (dataset, pixel_array)."""
    ds = pydicom.dcmread(path, force=True)
    try:
        arr = ds.pixel_array
    except Exception:
        arr = None
    return ds, arr


def pixel_spacing(ds, arr: Array) -> Tuple[float, float]:
    """Размер пикселя (мм по X, мм по Y) из тега (0040,0303) Exposed Area.

    В наборе нет PixelSpacing, но Exposed Area хранит физический размер снятой области
    в миллиметрах. Деление на размер кадра даёт реальный масштаб каждого снимка, поэтому
    углы и сантиметры считаются по факту, а не по предположению.
    """
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
    lo, hi = C.PIXEL_SPACING_LIMITS
    if not (lo <= sx <= hi and lo <= sy <= hi):
        return C.PIXEL_SPACING_MM
    return sx, sy


def dedupe_unique_images(files: List[str]) -> List[Tuple[str, object, Array]]:
    """Уникальные изображения исследования (по содержимому пикселей).

    Возвращает список (representative_path, dataset, array) в порядке
    возрастания InstanceNumber.
    """
    seen = {}
    for path in files:
        ds, arr = read_dicom(path)
        if arr is None:
            continue
        arr = np.asarray(arr)
        key = (arr.shape, arr.tobytes())
        if key not in seen:
            seen[key] = (path, ds, arr)
    items = list(seen.values())

    def _inst(t):
        try:
            return int(t[1].InstanceNumber)
        except Exception:
            return 0

    items.sort(key=_inst)
    return items


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
    """Определить анатомическую область по геометрии изображения."""
    cols = arr.shape[1]
    if cols >= C.SPINE_MIN_COLUMNS:
        return C.REGION_SPINE
    return C.REGION_FEMUR_LEFT if bone_slope(arr) >= 0 else C.REGION_FEMUR_RIGHT


# --------------------------------------------------------------------------- #
# Предобработка под нейросеть
# --------------------------------------------------------------------------- #
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


def preprocess(arr: Array, size: int = C.IMAGE_SIZE) -> Array:
    """Нормировать, дополнить до квадрата и масштабировать.

    Возвращает float32 [0,1] формы (size, size).
    """
    import cv2

    img = normalize_uint8(arr)
    img = pad_to_square(img, fill=0)
    interp = cv2.INTER_AREA if img.shape[0] > size else cv2.INTER_LINEAR
    img = cv2.resize(img, (size, size), interpolation=interp)
    return img.astype(np.float32) / 255.0
