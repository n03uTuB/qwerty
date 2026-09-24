# -*- coding: utf-8 -*-
"""Робастная предобработка DICOM для DXA-QC (Шаг 1 ТЗ: «dicom_parser»).

Модуль дополняет `dicom_io.py` и НЕ меняет его публичный контракт:
`dicom_io.to_display_float` остаётся как есть, поэтому кэш PNG и признаки
(`features.py`, которые читают сырой DICOM) не ломаются. Новый конвейер
подключается флагом в `DXADataset` и в инференсе.

Что исправлено относительно текущего `dicom_io`:

1. **Сжатые пиксели.** Перед `pixel_array` вызывается `decompress()` — иначе
   JPEG / JPEG 2000 из сторонних выгрузок падают, хотя `pylibjpeg` установлен.
2. **Многокадровые и цветные файлы.** Корректно различаются по
   `SamplesPerPixel` / `NumberOfFrames`, а не «первый кадр».
3. **MONOCHROME1** инвертируется ДО перцентильной нормировки (эквивалентно,
   но без риска перепутать порядок).
4. **VOI LUT** применяется только когда окно реально задано, и через
   `pydicom.pixels.apply_voi_lut` (учитывает `VOILUTSequence`, а не только
   `WindowCenter`).
5. **Костное окно (bone window).** DXA — 8 бит, мягкие ткани занимают
   верхнюю половину гистограммы и «съедают» контраст кортикальной кости.
   Окно растягивает диапазон костной ткани, а фон обнуляется.
6. **Шумоподавление перед CLAHE.** `bilateralFilter` убирает зерно, которое
   CLAHE иначе усиливает; это стабилизирует локальные признаки (контур
   малого вертела, средняя линия столба).

Конвейер одинаков для обучения и инференса — это важно, т.к. в отчёте
команды (docs/metrics_improvement.md, п.3) уже была ошибка train/inference
skew на VOI-LUT.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pydicom

from . import config as C

try:  # opencv есть в requirements (opencv-python-headless)
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

Array = np.ndarray


# --------------------------------------------------------------------------- #
# Чтение пикселей
# --------------------------------------------------------------------------- #
def read_pixels(path: str, force: bool = True) -> Tuple[object, Optional[Array]]:
    """Прочитать DICOM и вернуть (dataset, ndarray) без LUT-преобразований."""
    ds = pydicom.dcmread(path, force=force)
    return ds, pixel_array(ds)


def pixel_array(ds) -> Optional[Array]:
    """Надёжно достать 2D-массив пикселей (сжатие, кадры, каналы)."""
    # 1) распаковать сжатые пиксели (JPEG / JPEG2000 / RLE) — pylibjpeg уже в deps
    try:
        if getattr(ds, "file_meta", None) is not None and \
                getattr(ds.file_meta, "TransferSyntaxUID", None) is not None:
            ts = str(ds.file_meta.TransferSyntaxUID)
            if not ts.endswith("1.2.840.10008.1.2") and "ImplicitVRLittleEndian" not in ts:
                ds.decompress()
    except Exception:
        pass
    try:
        arr = ds.pixel_array
    except Exception:
        return None
    return _to_single_frame(ds, np.asarray(arr))


def _to_single_frame(ds, arr: Array) -> Array:
    """Свести массив к одному 2D-кадру, корректно обработав цвет и кадры."""
    if arr.ndim == 2:
        return arr
    spp = int(getattr(ds, "SamplesPerPixel", 1) or 1)
    frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    if spp > 1 and arr.shape[-1] in (3, 4):        # цветное -> яркость
        return arr[..., :3].mean(axis=-1).astype(arr.dtype)
    if arr.ndim == 3:                              # (frames, H, W) -> первый
        return arr[0]
    if arr.ndim == 4:                              # (frames, H, W, C)
        return arr[0, ..., :3].mean(axis=-1).astype(arr.dtype)
    _ = frames  # noqa: F841 (оставлено для читаемости ветвления)
    return arr.reshape(arr.shape[-2], arr.shape[-1])


# --------------------------------------------------------------------------- #
# LUT и нормировка
# --------------------------------------------------------------------------- #
def apply_modality(ds, arr: Array) -> Array:
    """Rescale Slope/Intercept (модальное преобразование DICOM)."""
    f = np.asarray(arr).astype(np.float32)
    try:
        from pydicom.pixels import apply_modality_lut

        return apply_modality_lut(f, ds).astype(np.float32)
    except Exception:
        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        inter = float(getattr(ds, "RescaleIntercept", 0) or 0)
        return f * slope + inter if (slope != 1.0 or inter != 0.0) else f


def apply_voi(ds, arr: Array) -> Array:
    """VOI LUT / окно, если оно задано (у данных организатора его нет)."""
    f = np.asarray(arr).astype(np.float32)
    has_voi = (getattr(ds, "WindowCenter", None) is not None
               or getattr(ds, "WindowWidth", None) is not None
               or getattr(ds, "VOILUTSequence", None) is not None)
    if not has_voi:
        return f
    try:
        from pydicom.pixels import apply_voi_lut

        return apply_voi_lut(f, ds, index=0).astype(np.float32)
    except Exception:
        return f


def percentile_norm(f: Array, lo_p: float = 0.5, hi_p: float = 99.5,
                    invert: bool = False) -> Array:
    """Робастная перцентильная нормировка в [0,1] с опциональной инверсией."""
    f = np.asarray(f).astype(np.float32)
    lo, hi = float(np.percentile(f, lo_p)), float(np.percentile(f, hi_p))
    if hi - lo < 1e-6:
        hi = lo + 1.0
    out = np.clip((f - lo) / (hi - lo), 0.0, 1.0)
    if invert:
        out = 1.0 - out
    return out.astype(np.float32)


def to_display_float(ds, arr: Array) -> Array:
    """modality LUT -> VOI LUT -> MONOCHROME1 -> перцентильная нормировка."""
    f = apply_modality(ds, arr)
    f = apply_voi(ds, f)
    invert = str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1"
    return percentile_norm(f, 0.5, 99.5, invert=invert)


# --------------------------------------------------------------------------- #
# DXA-специфичное усиление кости
# --------------------------------------------------------------------------- #
def body_mask(img01: Array, thr_pct: float = 25.0) -> Array:
    """Маска тела: всё, что ярче фона (воздух/стол в DXA почти чёрный)."""
    f = np.asarray(img01, dtype=np.float32)
    if cv2 is not None:
        u8 = (np.clip(f, 0, 1) * 255).round().astype(np.uint8)
        _t, m = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m = m > 0
        if m.mean() > 0.98:            # Otsu «съел» всё (однородный кадр)
            m = f > np.percentile(f, thr_pct)
    else:
        m = f > np.percentile(f, thr_pct)
    return m


def bone_window(img01: Array, body: Optional[Array] = None,
                lo_p: float = 55.0, hi_p: float = 99.5) -> Array:
    """Окно, растягивающее диапазон костной ткани; фон обнуляется.

    Мягкие ткани в DXA занимают верхнюю половину яркостей, кортикальная кость —
    верхние перцентили. Растяжка от `lo_p` к `hi_p` внутри маски тела повышает
    контраст кости и контуров вертелов, не теряя губчатую часть.
    """
    f = np.asarray(img01, dtype=np.float32)
    if body is None:
        body = body_mask(f)
    vals = f[body] if body.any() else f.ravel()
    lo, hi = float(np.percentile(vals, lo_p)), float(np.percentile(vals, hi_p))
    if hi - lo < 1e-6:
        hi = lo + 1.0
    out = np.clip((f - lo) / (hi - lo), 0.0, 1.0)
    out[~body] = 0.0
    return out.astype(np.float32)


def denoise_clahe(img01: Array, clip: Optional[float] = None,
                  tile: Optional[int] = None, denoise: bool = True) -> Array:
    """CLAHE с предварительным bilateral-шумоподавлением.

    Шум усиливается CLAHE и делает локальные признаки (выступ вертела,
    средняя линия) нестабильными между прогонами. Bilateral убирает зерно,
    сохраняя края кости.
    """
    if cv2 is None:
        return np.clip(img01, 0.0, 1.0).astype(np.float32)
    clip = C.CLAHE_CLIP if clip is None else clip
    tile = C.CLAHE_TILE if tile is None else tile
    u8 = (np.clip(img01, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    if denoise:
        u8 = cv2.bilateralFilter(u8, d=5, sigmaColor=25, sigmaSpace=25)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tile), int(tile)))
    return (clahe.apply(u8).astype(np.float32) / 255.0)


# --------------------------------------------------------------------------- #
# Геометрия приведения к квадрату
# --------------------------------------------------------------------------- #
def pad_to_square(img: Array, fill: float = 0.0) -> Array:
    h, w = img.shape[:2]
    s = max(h, w)
    out = np.full((s, s), fill, dtype=img.dtype)
    y0, x0 = (s - h) // 2, (s - w) // 2
    out[y0:y0 + h, x0:x0 + w] = img
    return out


def to_square_resize(img01: Array, size: int) -> Array:
    """Квадрат-паддинг -> ресайз. Пропорции сохраняются, поэтому УГЛЫ не искажаются."""
    if cv2 is not None:
        u8 = (np.clip(img01, 0, 1) * 255).round().astype(np.uint8)
        u8 = pad_to_square(u8, 0)
        interp = cv2.INTER_AREA if u8.shape[0] > size else cv2.INTER_LINEAR
        u8 = cv2.resize(u8, (size, size), interpolation=interp)
        return u8.astype(np.float32) / 255.0
    # numpy-fallback (без opencv)
    img = pad_to_square(np.asarray(img01, np.float32), 0.0)
    idx = (np.linspace(0, img.shape[0] - 1, size)).round().astype(int)
    return img[np.ix_(idx, idx)].astype(np.float32)


def pad_scale(shape: Tuple[int, int], size: int) -> float:
    """Во сколько раз сжато изображение после pad_to_square+resize.

    Нужно, чтобы перевести координаты лендмарков (в квадрате `size`) обратно в
    пиксели оригинала и затем в мм: ``px_orig = coord_norm * size * pad_scale``.
    """
    h, w = shape
    return float(max(h, w)) / float(size)


# --------------------------------------------------------------------------- #
# Единая точка входа
# --------------------------------------------------------------------------- #
def preprocess_for_net(arr: Array, size: int = C.IMAGE_SIZE, ds=None,
                       mode: str = "bone", use_clahe: bool = True,
                       is_display: bool = False) -> Array:
    """Готовый вход сети: float32 [0,1], форма (size, size).

    mode: "bone" — костное окно + denoise+CLAHE (рекомендуется для лендмарков);
          "display" — как в dicom_io (совместимо со старым кэшем).
    """
    if ds is not None:
        img = to_display_float(ds, arr)
    elif is_display:
        img = np.asarray(arr, dtype=np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        img = np.clip(img, 0.0, 1.0)
    else:
        img = percentile_norm(np.asarray(arr, np.float32))

    if mode == "bone":
        img = bone_window(img, body_mask(img))
    if use_clahe:
        img = denoise_clahe(img, denoise=(mode == "bone"))
    return to_square_resize(img, size)
