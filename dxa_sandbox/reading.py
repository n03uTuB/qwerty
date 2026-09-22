"""Чтение DICOM и подготовка пикселей.

Обработанные ловушки:
- сжатые пиксели (JPEG / JPEG 2000 / RLE) — нужны плагины pylibjpeg;
- Modality LUT (RescaleSlope / RescaleIntercept или ModalityLUTSequence);
- окно (WindowCenter / WindowWidth), в том числе многозначное;
- MONOCHROME1 — инвертированная шкала (кость тёмная);
- цветные изображения (RGB, YBR, PALETTE COLOR) и многокадровые файлы;
- размер пикселя в разных тегах.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pydicom
from PIL import Image
from pydicom.dataset import Dataset
from pydicom.multival import MultiValue
from pydicom.pixels import apply_color_lut, apply_modality_lut

SPACING_TAGS = ("PixelSpacing", "ImagerPixelSpacing", "NominalScannedPixelSpacing")
_NOT_DICOM_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".txt", ".md", ".csv", ".json", ".html", ".py", ".pdf", ".zip",
}


def is_dicom_file(path: Path) -> bool:
    """DICOM Part 10: 128 байт преамбулы, затем 'DICM'."""
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except OSError:
        return False


def _looks_like_raw_dicom(path: Path) -> bool:
    """Файлы без преамбулы встречаются в выгрузках старых аппаратов — пробуем прочитать «силой»."""
    if path.suffix.lower() in _NOT_DICOM_SUFFIXES:
        return False
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True, specific_tags=["SOPClassUID"])
        return "SOPClassUID" in ds
    except Exception:
        return False


def iter_dicom_files(root: str | Path) -> Iterator[Path]:
    root = Path(root)
    candidates = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    for path in candidates:
        if is_dicom_file(path) or _looks_like_raw_dicom(path):
            yield path


def read(path: str | Path, pixels: bool = True) -> Dataset:
    return pydicom.dcmread(path, stop_before_pixels=not pixels, force=True)


@dataclass
class Spacing:
    row_mm: float  # расстояние между центрами соседних строк (по вертикали)
    col_mm: float  # расстояние между центрами соседних столбцов (по горизонтали)
    source: str  # из какого тега взято


def pixel_spacing(ds: Dataset) -> Spacing | None:
    for keyword in SPACING_TAGS:
        value = ds.get(keyword)
        if value:
            return Spacing(float(value[0]), float(value[1]), keyword)
    return None


def _first(value):
    if isinstance(value, (MultiValue, list, tuple)):
        return value[0] if len(value) else None
    return value


def dicom_window(ds: Dataset) -> tuple[float, float] | None:
    center, width = ds.get("WindowCenter"), ds.get("WindowWidth")
    if center is None or width is None:
        return None
    return float(_first(center)), float(_first(width))


def is_color(ds: Dataset) -> bool:
    return int(ds.get("SamplesPerPixel", 1)) > 1 or ds.get("PhotometricInterpretation") == "PALETTE COLOR"


def get_frame(ds: Dataset, frame: int = 0) -> np.ndarray:
    arr = ds.pixel_array  # здесь падает, если нет декодера для сжатия
    if int(ds.get("NumberOfFrames", 1) or 1) > 1:
        arr = arr[frame]
    return arr


def luminance(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _invert_stored(ds: Dataset, arr: np.ndarray) -> np.ndarray:
    bits = int(ds.get("BitsStored", arr.dtype.itemsize * 8))
    if int(ds.get("PixelRepresentation", 0)) == 0:
        return (2**bits - 1) - arr.astype(np.int64)
    return -arr.astype(np.int64)


def to_display(ds: Dataset, frame: int = 0, window: tuple[float, float] | None = None) -> np.ndarray:
    """uint8 для показа и рисования: (H, W) для серых, (H, W, 3) для цветных. Кость всегда светлая."""
    arr = get_frame(ds, frame)
    if is_color(ds):
        if ds.get("PhotometricInterpretation") == "PALETTE COLOR":
            arr = apply_color_lut(arr, ds)
        if arr.dtype == np.uint8:
            return arr
        top = 65535 if arr.max() > 255 else 255
        return np.clip(arr.astype(np.float64) / top * 255, 0, 255).astype(np.uint8)

    values = apply_modality_lut(arr, ds).astype(np.float64)
    window = window or dicom_window(ds)
    if window is None:
        lo, hi = np.percentile(values, (0.5, 99.5))
    else:
        center, width = window
        lo, hi = center - width / 2, center + width / 2
    if hi <= lo:
        hi = lo + 1
    out = np.clip((values - lo) / (hi - lo), 0, 1) * 255
    if ds.get("PhotometricInterpretation") == "MONOCHROME1":
        out = 255 - out
    return out.astype(np.uint8)


def to_analysis(ds: Dataset, frame: int = 0) -> np.ndarray:
    """float32 в единицах аппарата, где «больше = плотнее». Для цветных — яркость."""
    if is_color(ds):
        return luminance(to_display(ds, frame))
    arr = get_frame(ds, frame)
    if ds.get("PhotometricInterpretation") == "MONOCHROME1":
        arr = _invert_stored(ds, arr)
    return apply_modality_lut(arr, ds).astype(np.float32)


def extract_encapsulated_document(ds: Dataset, out_path: str | Path) -> Path | None:
    """Достаёт вложенный PDF / CDA (Encapsulated Document)."""
    if "EncapsulatedDocument" not in ds:
        return None
    data = bytes(ds.EncapsulatedDocument)
    length = ds.get("EncapsulatedDocumentLength")
    if length:
        data = data[: int(length)]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return out_path


def save_png(image: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
    return path
