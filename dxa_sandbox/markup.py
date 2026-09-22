"""Извлечение разметки аппарата.

Разметка DXA встречается в трёх видах:
1. overlay-слои DICOM (группы 6000–601E) — битовые маски поверх изображения;
2. цветные линии, вшитые в пиксели (Secondary Capture / скриншот отчёта);
3. числа и координаты в SR, PDF или приватных тегах — см. inspection.private_tags_report.
"""

from __future__ import annotations

import numpy as np
from pydicom.dataset import Dataset
from scipy import ndimage

from .reading import luminance

# ---------------------------------------------------------------- overlay


def overlay_groups(ds: Dataset) -> list[int]:
    return sorted(
        {
            tag.group
            for tag in ds.keys()
            if 0x6000 <= tag.group <= 0x601E and tag.group % 2 == 0 and tag.element == 0x3000
        }
    )


def overlay_mask(ds: Dataset, group: int = 0x6000) -> np.ndarray:
    """Маска overlay размером с изображение (с учётом OverlayOrigin, который считается с 1)."""
    ov_rows = int(ds[group, 0x0010].value)
    ov_cols = int(ds[group, 0x0011].value)
    data = np.frombuffer(bytes(ds[group, 0x3000].value), dtype=np.uint8)
    # биты упакованы от младшего к старшему; многокадровые overlay берём первым кадром
    bits = np.unpackbits(data, bitorder="little")[: ov_rows * ov_cols]
    ov = bits.reshape(ov_rows, ov_cols).astype(bool)

    origin = ds.get((group, 0x0050))
    r0, c0 = (int(origin.value[0]) - 1, int(origin.value[1]) - 1) if origin is not None else (0, 0)
    rows, cols = int(ds.Rows), int(ds.Columns)
    mask = np.zeros((rows, cols), dtype=bool)
    y0, x0 = max(r0, 0), max(c0, 0)
    y1, x1 = min(r0 + ov_rows, rows), min(c0 + ov_cols, cols)
    if y1 > y0 and x1 > x0:
        mask[y0:y1, x0:x1] = ov[y0 - r0 : y1 - r0, x0 - c0 : x1 - c0]
    return mask


# ---------------------------------------------------------------- цветная разметка

HUE_RANGES = {
    "red": [(0, 15), (345, 360)],
    "orange": [(15, 45)],
    "yellow": [(45, 75)],
    "green": [(75, 160)],
    "cyan": [(160, 200)],
    "blue": [(200, 260)],
    "magenta": [(260, 345)],
}


def rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Оттенок в градусах, насыщенность и яркость в [0, 1]."""
    x = rgb[..., :3].astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(-1), x.min(-1)
    delta = mx - mn
    s = np.where(mx > 0, delta / np.maximum(mx, 1e-6), 0.0)
    h = np.zeros_like(mx)
    nz = delta > 1e-6
    rm = nz & (mx == r)
    gm = nz & (mx == g) & ~rm
    bm = nz & ~rm & ~gm
    h[rm] = (60 * (g - b)[rm] / delta[rm]) % 360
    h[gm] = 60 * (b - r)[gm] / delta[gm] + 120
    h[bm] = 60 * (r - g)[bm] / delta[bm] + 240
    return h, s, mx


def color_markup_masks(
    rgb: np.ndarray, min_saturation: float = 0.4, min_value: float = 0.3, min_pixels: int = 20
) -> dict[str, np.ndarray]:
    """Маски цветной разметки по цветам. Серое изображение имеет насыщенность ~0, разметка — высокую."""
    h, s, v = rgb_to_hsv(rgb)
    colored = (s >= min_saturation) & (v >= min_value)
    masks = {}
    for name, ranges in HUE_RANGES.items():
        mask = np.zeros(colored.shape, dtype=bool)
        for lo, hi in ranges:
            mask |= (h >= lo) & (h < hi)
        mask &= colored
        if mask.sum() >= min_pixels:
            masks[name] = mask
    return masks


def components(mask: np.ndarray, min_pixels: int = 10) -> list[dict]:
    """Связные компоненты: bbox (x0, y0, x1, y1), число пикселей, центр (x, y)."""
    labels, _ = ndimage.label(mask)
    result = []
    for index, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        region = labels[sl] == index
        size = int(region.sum())
        if size < min_pixels:
            continue
        ys, xs = np.nonzero(region)
        result.append(
            {
                "bbox": (sl[1].start, sl[0].start, sl[1].stop, sl[0].stop),
                "pixels": size,
                "centroid": (float(xs.mean() + sl[1].start), float(ys.mean() + sl[0].start)),
            }
        )
    return result


def horizontal_line_rows(mask: np.ndarray, min_fraction: float = 0.5) -> list[float]:
    """Y-координаты горизонтальных линий (например, межпозвонковых границ на скриншоте отчёта).

    min_fraction — минимальная длина линии как доля от самой длинной.
    """
    counts = mask.sum(axis=1)
    if counts.max() == 0:
        return []
    rows = np.nonzero(counts >= counts.max() * min_fraction)[0]
    groups = np.split(rows, np.nonzero(np.diff(rows) > 1)[0] + 1)
    return [float(g.mean()) for g in groups if len(g)]


def remove_markup(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Серое изображение без разметки: пиксели разметки заменяются ближайшими «чистыми»."""
    gray = luminance(rgb)
    if not mask.any():
        return gray.astype(np.uint8)
    grown = ndimage.binary_dilation(mask, iterations=1)  # края линий со сглаживанием
    _, (iy, ix) = ndimage.distance_transform_edt(grown, return_indices=True)
    return gray[iy, ix].astype(np.uint8)
