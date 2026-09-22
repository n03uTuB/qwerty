# -*- coding: utf-8 -*-
"""Геометрические признаки для контроля качества (гибридный подход).

Идея: часть критериев качества — измеримые геометрические величины:
  * ось позвоночника (отклонение в градусах);
  * наклон/ориентация проксимального отдела бедра (ротация).

Эти признаки вычисляются детерминированно и дополняют нейросетевые
вероятности (feature-fusion на уровне решения).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import config as C

FEATURES_CSV = os.path.join(C.ARTIFACTS_DIR, "features.csv")

# Набор геометрических признаков (имена фиксированы)
FEATURE_NAMES = ["spine_axis_angle", "spine_midline_angle", "femur_axis_angle",
                 "bone_eccentricity", "bright_area_ratio", "vertical_symmetry"]


def _principal_axis(mask: np.ndarray, spacing=(1.0, 1.0)):
    """Главная ось бинарной маски: (угол_к_вертикали_град, эксцентриситет).

    ``spacing`` — физический размер пикселя (x, y) в мм; координаты
    масштабируются, поэтому угол и эксцентриситет получаются физическими
    (важно для критерия «отклонение оси > 5°»).
    """
    ys, xs = np.where(mask)
    if len(xs) < 10:
        return 0.0, 0.0
    sx, sy = spacing
    x = xs.astype(np.float64) * sx
    y = ys.astype(np.float64) * sy
    x -= x.mean()
    y -= y.mean()
    cov = np.array([[np.mean(x * x), np.mean(x * y)],
                    [np.mean(x * y), np.mean(y * y)]])
    vals, vecs = np.linalg.eigh(cov)
    main = vecs[:, int(np.argmax(vals))]
    if main[1] < 0:
        main = -main
    # угол между главной осью и вертикалью (ось Y)
    angle = np.degrees(np.arctan2(abs(main[0]), abs(main[1]) + 1e-9))
    ecc = float(np.sqrt(max(vals.max(), 1e-9) / max(vals.min(), 1e-9)))
    return float(angle), ecc


def spine_axis_angle(arr: np.ndarray, thr_pct: float = 90.0,
                     spacing=C.PIXEL_SPACING_MM) -> float:
    """Оценка отклонения оси позвоночника от вертикали (в градусах).

    Позвоночник выделяется как яркая компактная структура; вычисляется угол
    главной оси. Возвращает угол в градусах (0 — строго вертикально).
    """
    f = arr.astype(np.float32)
    # ограничиваем центральной областью (исключаем подвздошные кости по краям)
    h, w = f.shape
    x0, x1 = int(w * 0.15), int(w * 0.85)
    roi = f[:, x0:x1]
    if roi.max() <= roi.min():
        return 0.0
    thr = np.percentile(roi, thr_pct)
    mask = roi >= thr
    angle, _ = _principal_axis(mask, spacing)
    return angle


def spine_midline_angle(arr: np.ndarray, spacing=C.PIXEL_SPACING_MM,
                        level: float = 0.25) -> float:
    """Наклон оси по средней линии позвоночного столба (градусы к вертикали).

    Главная ось всей яркой области занижает наклон: её «тянет» вертикальная протяжённость
    снимка. Врач же смотрит на линию, проходящую через центры тел позвонков. На наборе
    организатора этот признак отделяет метку «ось не выровнена» заметно лучше
    (AUC 0.83 против 0.73 у spine_axis_angle), а его значения совпадают со шкалой ТЗ:
    медиана 5.0° при нарушении против 2.5° в норме (в ТЗ примеры 6.9° и 1.6°).
    """
    from scipy import ndimage

    f = arr.astype(np.float32)
    rows, cols = f.shape
    band = f[: int(rows * 0.8), int(cols * 0.2): int(cols * 0.8)]
    if band.size == 0 or band.max() <= band.min():
        return 0.0
    # порог между мягкими тканями и костью (перцентильный порог режет губчатую часть)
    soft, dense = np.percentile(band, (50, 98))
    if dense <= soft:
        return 0.0
    mask = band > soft + (dense - soft) * level
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(mask, np.ones((5, 5))))
    centers, ys = [], []
    for row in range(mask.shape[0]):
        xs = np.nonzero(mask[row])[0]
        if len(xs) < 3:
            continue
        groups = np.split(xs, np.nonzero(np.diff(xs) > 1)[0] + 1)
        longest = max(groups, key=len)
        if len(longest) < 3:
            continue
        centers.append(float(longest.mean()))
        ys.append(float(row))
    if len(centers) < 20:
        return 0.0
    slope, _ = np.polyfit(np.array(ys), np.array(centers), 1)
    sx, sy = spacing
    return float(np.degrees(np.arctan2(abs(slope) * sx, sy)))


def femur_axis_angle(arr: np.ndarray, thr_pct: float = 92.0,
                     spacing=C.PIXEL_SPACING_MM) -> float:
    """Наклон главной оси бедренной кости (в градусах от вертикали)."""
    f = arr.astype(np.float32)
    if f.max() <= f.min():
        return 0.0
    thr = np.percentile(f, thr_pct)
    mask = f >= thr
    angle, _ = _principal_axis(mask, spacing)
    return angle


def bone_eccentricity(arr: np.ndarray, thr_pct: float = 92.0,
                      spacing=C.PIXEL_SPACING_MM) -> float:
    """Вытянутость костной структуры (отношение главных дисперсий)."""
    f = arr.astype(np.float32)
    if f.max() <= f.min():
        return 0.0
    thr = np.percentile(f, thr_pct)
    _angle, ecc = _principal_axis(f >= thr, spacing)
    return ecc


def bright_area_ratio(arr: np.ndarray, thr_pct: float = 92.0) -> float:
    """Доля площади изображения, занятая яркой (костной) структурой."""
    f = arr.astype(np.float32)
    if f.max() <= f.min():
        return 0.0
    thr = np.percentile(f, thr_pct)
    return float((f >= thr).mean())


def vertical_symmetry(arr: np.ndarray) -> float:
    """Асимметрия распределения костной структуры по горизонтали (0..1)."""
    f = arr.astype(np.float32)
    if f.max() <= f.min():
        return 0.0
    thr = np.percentile(f, 90)
    mask = (f >= thr).astype(np.float32)
    col = mask.sum(0)
    if col.sum() <= 0:
        return 0.0
    left = col[: len(col) // 2].sum()
    right = col[len(col) // 2:].sum()
    return float(abs(left - right) / (col.sum() + 1e-9))


def compute_features(arr: np.ndarray, spacing=C.PIXEL_SPACING_MM) -> dict:
    """Полный вектор геометрических признаков для изображения.

    spacing — реальный размер пикселя снимка (dicom_io.pixel_spacing). От него зависят
    все углы: при ошибочном соотношении сторон наклон оси занижается в 1.75 раза.
    """
    return {
        "spine_axis_angle": spine_axis_angle(arr, spacing=spacing),
        "spine_midline_angle": spine_midline_angle(arr, spacing=spacing),
        "femur_axis_angle": femur_axis_angle(arr, spacing=spacing),
        "bone_eccentricity": bone_eccentricity(arr, spacing=spacing),
        "bright_area_ratio": bright_area_ratio(arr),
        "vertical_symmetry": vertical_symmetry(arr),
    }


def build_features(manifest_csv: str = C.MANIFEST_CSV,
                   out_csv: str = FEATURES_CSV, verbose: bool = True) -> pd.DataFrame:
    """Вычислить геометрические признаки для всех изображений манифеста."""
    from PIL import Image

    df = pd.read_csv(manifest_csv)
    rows = []
    for _, r in df.iterrows():
        arr = np.asarray(Image.open(r["cache_path"]).convert("L"))
        spacing = (float(r["spacing_x"]), float(r["spacing_y"])) \
            if "spacing_x" in df.columns and pd.notna(r.get("spacing_x")) else C.PIXEL_SPACING_MM
        feat = compute_features(arr, spacing)
        feat["image_uid"] = r["image_uid"]
        feat["study_uid"] = r["study_uid"]
        rows.append(feat)
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False, encoding="utf-8")
    if verbose:
        print(f"[features] сохранено: {out_csv}  строк: {len(out)}")
        print(out[FEATURE_NAMES].describe().to_string())
    return out


def load_features(manifest: pd.DataFrame,
                  features_csv: str = FEATURES_CSV) -> pd.DataFrame:
    """Присоединить геометрические признаки к манифесту (по image_uid)."""
    if not os.path.isfile(features_csv):
        build_features(out_csv=features_csv)
    feat = pd.read_csv(features_csv)
    return manifest.merge(feat[["image_uid"] + FEATURE_NAMES], on="image_uid",
                          how="left")
