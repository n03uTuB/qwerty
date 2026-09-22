"""Проверки качества по изображению — ДЕМО на синтетическом фантоме.

Классические правила показывают устройство проверок: найти анатомию → сравнить с нормой и с разметкой.
На реальных снимках поиск анатомии заменяется моделями (сегментация позвонков, ключевые точки бедра),
а сравнение и пороги остаются.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from .geometry import image_center, otsu_threshold, rotate_image, rotate_points, spine_axis_tilt
from .markup import components
from .qc import Finding

TILT_LIMIT_DEG = 5.0  # демо-пороги, согласовать с экспертами
NOISE_LIMIT = 0.08
METAL_FACTOR = 1.4
METAL_MAX_AREA = 0.01
MIN_VERTEBRAE = 5


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _rectangle(bbox, pad: float = 0.0) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = bbox
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]


def bone_mask(values: np.ndarray) -> np.ndarray:
    return values > otsu_threshold(values)


# ---------------------------------------------------------------- любые снимки


def check_foreign_body(values: np.ndarray) -> list[Finding]:
    """Металл намного плотнее кости: компактные объекты выше диапазона костной плотности."""
    p50, p99 = np.percentile(values, (50, 99))
    if p99 <= p50:
        return []
    threshold = p50 + (p99 - p50) * METAL_FACTOR
    found = [c for c in components(values > threshold, min_pixels=6) if c["pixels"] <= values.size * METAL_MAX_AREA]
    if not found:
        return []
    return [Finding(
        "foreign_body", f"Артефакт от инородного тела: объектов {len(found)}", "warning",
        "Компактные объекты плотнее кости", defect="foreign_body_artifact", confidence=0.95,
        polylines=[_rectangle(c["bbox"], pad=3) for c in found],
    )]


def check_acquisition_noise(values: np.ndarray) -> list[Finding]:
    """Шум относительно контраста кость/мягкие ткани: признак неверных физико-технических параметров."""
    residual = values - ndimage.median_filter(values, size=3)
    sigma = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
    p50, p99 = np.percentile(values, (50, 99))
    ratio = sigma / max(float(p99 - p50), 1e-6)
    violation = ratio > NOISE_LIMIT
    return [Finding(
        "noise_to_contrast", f"Шум/контраст {ratio:.3f}", "warning" if violation else "info",
        f"Допуск {NOISE_LIMIT} (демо-порог)", value=round(ratio, 4),
        defect="acquisition_parameters", confidence=round(_sigmoid((ratio - NOISE_LIMIT) / 0.02), 3),
    )]


# ---------------------------------------------------------------- позвоночник


def check_spine_tilt(values: np.ndarray) -> tuple[list[Finding], float]:
    angle, axis = spine_axis_tilt(values)
    if angle is None:
        return [], 0.0
    violation = abs(angle) > TILT_LIMIT_DEG
    return [Finding(
        "spine_axis_tilt", f"Наклон оси позвоночника {angle:+.1f}°", "warning" if violation else "info",
        f"Допуск ±{TILT_LIMIT_DEG:g}° (демо-порог)", value=round(angle, 2), unit="deg", polylines=[axis],
        defect="positioning", confidence=round(_sigmoid((abs(angle) - TILT_LIMIT_DEG) * 1.5), 3),
    )], angle


def spine_segments(mask: np.ndarray, band=(0.3, 0.7), min_fill: float = 0.3) -> list[tuple[int, int]]:
    """Тела позвонков как вертикальные отрезки (y0, y1) в центральной полосе выпрямленного снимка."""
    rows, cols = mask.shape
    filled = mask[:, int(cols * band[0]) : int(cols * band[1])].mean(axis=1) >= min_fill
    idx = np.nonzero(filled)[0]
    if not len(idx):
        return []
    groups = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
    return [(int(g[0]), int(g[-1])) for g in groups if len(g) >= max(3, rows * 0.03)]


def dense_tissue_mask(values: np.ndarray, level: float = 0.25) -> np.ndarray:
    """Более низкий порог, чем у тел позвонков: попадают и рёбра, и гребни подвздошных костей."""
    soft, bone = np.percentile(values, (50, 99))
    return values > soft + (bone - soft) * level


def rib_segment_index(dense: np.ndarray, segments, lateral=((0.08, 0.3), (0.7, 0.92)), min_fill: float = 0.05) -> int | None:
    """Индекс T12 — самого верхнего позвонка с рёбрами с обеих сторон (dense — маска dense_tissue_mask)."""
    rows, cols = dense.shape
    for index, (y0, y1) in enumerate(segments):
        if (y0 + y1) / 2 > rows * 0.5:
            break
        region = dense[y0 : y1 + 1]
        fills = [region[:, int(cols * a) : int(cols * b)].mean() for a, b in lateral]
        if min(fills) >= min_fill:
            return index
    return None


def check_spine_structure(values: np.ndarray, markup_masks: dict[str, np.ndarray], angle: float) -> list[Finding]:
    """Охват кадра, межпозвонковые границы и нумерация разметки аппарата."""
    rows, cols = values.shape
    mask, dense = bone_mask(values), dense_tissue_mask(values)
    rotate = abs(angle) > 0.5
    center = image_center(values.shape)
    if rotate:  # выпрямляем снимок, чтобы позвонки и промежутки шли по строкам
        mask = rotate_image(mask.astype(np.float32), -angle, order=0) > 0.5
        dense = rotate_image(dense.astype(np.float32), -angle, order=0) > 0.5
        markup_masks = {k: rotate_image(m.astype(np.float32), -angle, order=0) > 0.5 for k, m in markup_masks.items()}

    def back(points):
        return rotate_points(points, angle, center) if rotate else points

    findings = []
    segments = spine_segments(mask)
    touching = [s for s in segments if s[0] <= 1 or s[1] >= rows - 2]
    if touching or len(segments) < MIN_VERTEBRAE:
        lines = [back([(0, s[0] if s[0] <= 1 else s[1]), (cols - 1, s[0] if s[0] <= 1 else s[1])]) for s in touching]
        title = "Позвонки обрезаны краем снимка" if touching else f"В кадре мало позвонков: {len(segments)}"
        findings.append(Finding("incomplete_coverage", title, "warning",
                                f"Найдено тел позвонков: {len(segments)}", defect="incomplete_coverage",
                                confidence=0.9, polylines=lines))
    if len(segments) < 3:
        return findings

    tolerance = max(6.0, 0.25 * float(np.median([s[1] - s[0] for s in segments])))
    gaps = [(a[1] + b[0]) / 2 for a, b in zip(segments, segments[1:])]
    bounds = [float(segments[0][0])] + gaps + [float(segments[-1][1])]

    green = markup_masks.get("green")
    if green is not None and gaps:
        ys = [c["centroid"][1] for c in components(green, min_pixels=15)]
        bad = [y for y in ys if min(abs(y - g) for g in gaps) > tolerance]
        if bad:
            findings.append(Finding(
                "boundary_misplaced", f"Межпозвонковые границы вне промежутков: {len(bad)}", "warning",
                f"Допуск {tolerance:.0f} пикс.", defect="roi_placement", confidence=0.85,
                polylines=[back([(cols * 0.3, y), (cols * 0.7, y)]) for y in bad],
            ))

    red = markup_masks.get("red")
    t12 = rib_segment_index(dense, segments)
    if red is not None and t12 is not None and t12 + 1 < len(bounds):
        # рамка ROI может распасться на куски (её перекрывают другие линии) — берём всю красную маску
        ys, xs = np.nonzero(red)
        if len(ys):
            x0, top, x1 = int(xs.min()), int(ys.min()), int(xs.max())
            distances = [abs(top - b) for b in bounds]
            nearest = int(np.argmin(distances))
            shift = nearest - (t12 + 1)
            if shift and distances[nearest] <= tolerance * 1.5:
                findings.append(Finding(
                    "vertebra_labeling", f"Нумерация позвонков сдвинута на {shift:+d} уровень", "error",
                    "L1 определён как позвонок под последним позвонком с рёбрами (T12)", value=shift,
                    defect="vertebra_labeling", confidence=0.85, polylines=[back([(x0, top), (x1, top)])],
                ))
    return findings


def image_checks(values: np.ndarray, markup_masks: dict[str, np.ndarray], kind: str | None) -> list[Finding]:
    findings = check_foreign_body(values) + check_acquisition_noise(values)
    if kind == "spine":
        tilt, angle = check_spine_tilt(values)
        findings += tilt
        findings += check_spine_structure(values, markup_masks, angle)
    # Бедро: ротация, охват и положение ROI шейки требуют модели ключевых точек — см. README.
    return findings
