# -*- coding: utf-8 -*-
"""Предобработка пикселей DXA (порт конвейера fable_solution).

Два уровня:
  1. :func:`to_display_float` / :func:`to_display_uint8` — приведение «сырого»
     массива DICOM к видимому виду (modality LUT + VOI LUT + инверсия
     MONOCHROME1 + перцентильная нормировка). Этот же вид кэшируется в PNG и
     используется при инференсе — train/inference не расходятся.
  2. :func:`preprocess_for_net` — вход сети: костное окно (bone window),
     двусторонний denoise, CLAHE, pad-to-square, resize и нормализация.

Всё на numpy/cv2, без albumentations — конвейер детерминирован и одинаков
в обучении и в сервисе.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from . import config as C


# --------------------------------------------------------------------------- #
# Уровень 1: приведение DICOM к «дисплейному» виду
# --------------------------------------------------------------------------- #
def _to_float(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32)


def _apply_modality_lut(ds, arr: np.ndarray) -> np.ndarray:
    """Modality LUT: RescaleSlope/Intercept (иначе — без изменений)."""
    out = _to_float(arr)
    if ds is None:
        return out
    slope = getattr(ds, "RescaleSlope", None)
    intercept = getattr(ds, "RescaleIntercept", None)
    try:
        slope = float(slope) if slope is not None else 1.0
        intercept = float(intercept) if intercept is not None else 0.0
    except (TypeError, ValueError):
        slope, intercept = 1.0, 0.0
    if slope not in (0.0, 1.0) or intercept != 0.0:
        out = out * slope + intercept
    return out


def _apply_voi_lut(ds, arr: np.ndarray) -> np.ndarray:
    """VOI LUT: окно (WindowCenter/Width) либо pydicom.apply_voi_lut, иначе норм."""
    out = _to_float(arr)
    if ds is None:
        return out
    try:
        import pydicom

        if getattr(ds, "WindowCenter", None) is not None or \
                getattr(ds, "VOILUTSequence", None) is not None:
            out = _to_float(pydicom.pixel_data_handlers.util.apply_voi_lut(
                arr, ds, index=0))
            return out
    except Exception:
        pass
    return out


def _percentile_norm(arr: np.ndarray, low: float = 1.0,
                     high: float = 99.0) -> np.ndarray:
    """Перцентильная нормировка к [0, 1] — устойчива к выбросам и металлу."""
    out = _to_float(arr)
    lo, hi = np.percentile(out, [low, high])
    if hi <= lo:
        lo, hi = float(out.min()), float(out.max())
    if hi <= lo:
        return np.zeros_like(out, dtype=np.float32)
    out = (out - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def to_display_float(ds, arr: np.ndarray, low: float = 1.0,
                     high: float = 99.0) -> np.ndarray:
    """Сырой массив DICOM -> float32 [0, 1] в «дисплейном» виде."""
    out = _apply_modality_lut(ds, arr)
    out = _apply_voi_lut(ds, out)
    out = _percentile_norm(out, low=low, high=high)
    # MONOCHROME1: белое — это низкие значения, инвертируем.
    if ds is not None and getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        out = 1.0 - out
    return out.astype(np.float32)


def to_display_uint8(ds, arr: np.ndarray) -> np.ndarray:
    """То же, но uint8 [0, 255] — для PNG-кэша."""
    return np.clip(to_display_float(ds, arr) * 255.0, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Уровень 2: вход сети
# --------------------------------------------------------------------------- #
def bone_window(img01: np.ndarray, lo: float = 0.15,
                hi: float = 0.95) -> np.ndarray:
    """Костное окно: растянуть интересующий диапазон, обрезать фон/мягкие ткани.

    Для DXA полезный сигнал (кость) лежит в верхней части яркостного диапазона;
    линейное растяжение [lo, hi] -> [0, 1] повышает контраст трабекул.
    """
    out = (np.asarray(img01, dtype=np.float32) - lo) / max(hi - lo, 1e-6)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def denoise(img8: np.ndarray) -> np.ndarray:
    """Двусторонний фильтр — убирает зерно, сохраняя края кости."""
    import cv2

    return cv2.bilateralFilter(img8, d=5, sigmaColor=35, sigmaSpace=35)


def apply_clahe(img8: np.ndarray, clip: float = C.CLAHE_CLIP,
                tile: int = C.CLAHE_TILE) -> np.ndarray:
    """CLAHE — выравнивание локального контраста."""
    import cv2

    clahe = cv2.createCLAHE(clipLimit=float(clip),
                            tileGridSize=(int(tile), int(tile)))
    return clahe.apply(img8)


def pad_to_square(img: np.ndarray) -> np.ndarray:
    """Дополнить до квадрата (replicate) — без искажения пропорций анатомии."""
    h, w = img.shape[:2]
    if h == w:
        return img
    s = max(h, w)
    top = (s - h) // 2
    bottom = s - h - top
    left = (s - w) // 2
    right = s - w - left
    pad = [(top, bottom), (left, right)] + [(0, 0)] * (img.ndim - 2)
    return np.pad(img, pad, mode="edge")


def preprocess_for_net(display: np.ndarray, mode: Optional[str] = None,
                       size: Optional[int] = None,
                       mean: float = 0.5, std: float = 0.25) -> np.ndarray:
    """Дисплейное изображение -> тензор-массив (1, size, size) float32.

    ``mode``: ``"bone"`` (костное окно + denoise + CLAHE) или ``"display"``
    (как базовое решение: только нормировка + CLAHE).
    """
    import cv2

    mode = mode or C.PREPROCESS_MODE
    size = int(size or C.IMAGE_SIZE)
    img = np.asarray(display, dtype=np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)

    if mode == "bone" and C.USE_BONE_WINDOW:
        img = bone_window(img)

    img8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    if C.USE_DENOISE:
        img8 = denoise(img8)
    if C.USE_CLAHE:
        img8 = apply_clahe(img8)

    img8 = pad_to_square(img8)
    interp = cv2.INTER_AREA if img8.shape[0] > size else cv2.INTER_LINEAR
    img8 = cv2.resize(img8, (size, size), interpolation=interp)

    out = img8.astype(np.float32) / 255.0
    out = (out - mean) / std
    return out[None, :, :].astype(np.float32)


def preprocess_for_net_uint8(display: np.ndarray, mode: Optional[str] = None,
                             size: Optional[int] = None) -> np.ndarray:
    """То же, но без финальной нормализации — uint8 (для визуализации/аудита)."""
    x = preprocess_for_net(display, mode=mode, size=size, mean=0.0, std=1.0)
    return np.clip(x[0] * 255.0, 0, 255).astype(np.uint8)