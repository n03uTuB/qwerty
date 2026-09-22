"""Геометрия для правил качества: пороги, оси, углы, перевод в миллиметры."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .reading import Spacing


def image_center(shape: tuple[int, ...]) -> tuple[float, float]:
    rows, cols = shape[:2]
    return (cols - 1) / 2, (rows - 1) / 2


def rotate_image(image: np.ndarray, angle_deg: float, order: int = 1) -> np.ndarray:
    """Поворот против часовой стрелки на экране вокруг центра, размер сохраняется."""
    return ndimage.rotate(image, angle_deg, reshape=False, order=order, mode="nearest")


def rotate_points(points, angle_deg: float, center: tuple[float, float]) -> list[tuple[float, float]]:
    """Поворот точек (x, y) согласованно с rotate_image."""
    theta = np.radians(angle_deg)
    c, s = float(np.cos(theta)), float(np.sin(theta))
    cx, cy = center
    return [(cx + (x - cx) * c + (y - cy) * s, cy - (x - cx) * s + (y - cy) * c) for x, y in points]


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    hist, edges = np.histogram(values.ravel(), bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    weight1 = np.cumsum(hist)
    weight2 = weight1[-1] - weight1
    mean1 = np.cumsum(hist * centers) / np.maximum(weight1, 1)
    mean2 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(weight2, 1)
    between = weight1 * weight2 * (mean1 - mean2) ** 2
    return float(centers[int(np.argmax(between))])


def principal_axis(mask: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Центр (x, y) и единичный вектор главной оси маски, направленный вниз по изображению."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return None
    coords = np.stack([xs, ys]).astype(np.float64)
    center = coords.mean(axis=1)
    eigvals, eigvecs = np.linalg.eigh(np.cov(coords))
    vx, vy = eigvecs[:, int(np.argmax(eigvals))]
    if vy < 0:
        vx, vy = -vx, -vy
    return (float(center[0]), float(center[1])), (float(vx), float(vy))


def angle_from_vertical(direction: tuple[float, float]) -> float:
    """Угол в градусах; плюс — нижний конец оси смещён вправо на изображении."""
    vx, vy = direction
    return float(np.degrees(np.arctan2(vx, vy)))


def spine_axis_tilt(gray: np.ndarray) -> tuple[float | None, list[tuple[float, float]]]:
    """ДЕМО: наклон оси позвоночника по яркой центральной части снимка.

    На реальных данных вместо порога Оцу должна стоять сегментация позвонков.
    """
    rows, cols = gray.shape
    x_offset = int(cols * 0.2)
    roi = gray[: int(rows * 0.75), x_offset : int(cols * 0.8)].astype(np.float64)
    mask = roi > otsu_threshold(roi)
    axis = principal_axis(mask)
    if axis is None:
        return None, []
    (cx, cy), (vx, vy) = axis
    cx += x_offset
    half = rows * 0.4
    line = [(cx - vx * half, cy - vy * half), (cx + vx * half, cy + vy * half)]
    return angle_from_vertical((vx, vy)), line


def distance_mm(p0: tuple[float, float], p1: tuple[float, float], spacing: Spacing) -> float:
    """Расстояние между точками (x, y) в мм с учётом разного шага по строкам и столбцам."""
    dx = (p1[0] - p0[0]) * spacing.col_mm
    dy = (p1[1] - p0[1]) * spacing.row_mm
    return float(np.hypot(dx, dy))
