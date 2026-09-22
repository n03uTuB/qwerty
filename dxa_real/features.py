"""Геометрические признаки ровно по критериям ТЗ.

Позвоночник:
  * ось позвоночника — угол к вертикали (допуск 5°);
  * охват кадра — видны ли гребни подвздошных костей снизу и Th12 с рёбрами сверху;
  * посторонние предметы — компактные объекты плотнее кости.
Бедро:
  * отступы поля сканирования (в ТЗ: 3 см сверху и снизу, 2 см сбоку);
  * ротация — выраженность малого вертела и угол диафиза.
Признаки детерминированы и объяснимы: каждый можно показать врачу на снимке.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .data import PIXEL_SPACING_MM, Image

Spacing = tuple[float, float]  # (мм по X, мм по Y) — берётся из Exposed Area каждого снимка


def _otsu(values: np.ndarray, bins: int = 256) -> float:
    hist, edges = np.histogram(values.ravel(), bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    weight1 = np.cumsum(hist)
    weight2 = weight1[-1] - weight1
    cumulative = np.cumsum(hist * centers)
    mean1 = cumulative / np.maximum(weight1, 1)
    mean2 = (cumulative[-1] - cumulative) / np.maximum(weight2, 1)
    return float(centers[int(np.argmax(weight1 * weight2 * (mean1 - mean2) ** 2))])


def normalize(array: np.ndarray) -> np.ndarray:
    values = array.astype(np.float32)
    lo, hi = np.percentile(values, (0.5, 99.5))
    return np.clip((values - lo) / max(hi - lo, 1e-6), 0, 1)


SPINE_LEVEL = 0.25   # подобрано визуально: контур позвонков и гребней без мягких тканей
FEMUR_LEVEL = 0.15   # вся бедренная кость целиком, включая губчатую часть шейки


def bone_mask(values: np.ndarray, level: float = 0.2) -> np.ndarray:
    """Кость целиком, а не только плотные края.

    Порог Оцу на DXA отсекает губчатую часть и даёт рваный контур, поэтому берём
    долю между медианой мягких тканей и плотной костью, затем заливаем дыры.
    """
    soft, dense = np.percentile(values, (50, 98))
    mask = values > soft + (dense - soft) * level
    mask = ndimage.binary_closing(mask, np.ones((5, 5)))
    mask = ndimage.binary_fill_holes(mask)
    return ndimage.binary_opening(mask, np.ones((3, 3)))


def largest_component(mask: np.ndarray, prefer_bottom: bool = False) -> np.ndarray:
    """Самая крупная связная кость; с prefer_bottom — та, что доходит до низа кадра
    (для бедра это диафиз, что отделяет бедренную кость от таза)."""
    labels, count = ndimage.label(mask)
    if count <= 1:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    if prefer_bottom:
        bottom = set(np.unique(labels[-3:])) - {0}
        candidates = [i for i in bottom if sizes[i] > mask.size * 0.01]
        if candidates:
            return labels == max(candidates, key=lambda i: sizes[i])
    return labels == int(np.argmax(sizes))


def _principal_axis(mask: np.ndarray, spacing: Spacing) -> tuple[float, float]:
    """Угол главной оси к вертикали в градусах (в физических мм) и вытянутость."""
    mm_x, mm_y = spacing
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return 0.0, 0.0
    coords = np.stack([xs * mm_x, ys * mm_y]).astype(np.float64)
    eigvals, eigvecs = np.linalg.eigh(np.cov(coords))
    vx, vy = eigvecs[:, int(np.argmax(eigvals))]
    angle = float(np.degrees(np.arctan2(abs(vx), abs(vy) + 1e-9)))
    ecc = float(np.sqrt(max(eigvals.max(), 1e-9) / max(eigvals.min(), 1e-9)))
    return angle, ecc


def _edge_fraction(mask: np.ndarray, side: str, band: int = 2) -> float:
    if side == "top":
        strip = mask[:band]
    elif side == "bottom":
        strip = mask[-band:]
    elif side == "left":
        strip = mask[:, :band]
    else:
        strip = mask[:, -band:]
    return float(strip.mean())


def _margin_cm(mask: np.ndarray, side: str, spacing: Spacing) -> float:
    """Расстояние от кости до края кадра в сантиметрах."""
    mm_x, mm_y = spacing
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return 0.0
    rows, cols = mask.shape
    if side == "top":
        return float(ys.min() * mm_y / 10)
    if side == "bottom":
        return float((rows - 1 - ys.max()) * mm_y / 10)
    if side == "left":
        return float(xs.min() * mm_x / 10)
    return float((cols - 1 - xs.max()) * mm_x / 10)


def foreign_body(values: np.ndarray, level: float = 0.75, max_area: float = 0.03) -> dict:
    """Посторонние предметы: яркие объекты ВНЕ костных структур.

    Снимки пересвечены — плотная кость упирается в максимум яркости, поэтому признак
    «ярче кости» не работает. Металл от одежды отличается расположением: он лежит
    в мягких тканях и не связан с позвоночником или тазом.
    """
    empty = dict(foreign_count=0.0, foreign_area=0.0, foreign_max_area=0.0,
                 foreign_compactness=0.0, foreign_top=0.0)
    soft, dense = np.percentile(values, (50, 98))
    if dense <= soft:
        return empty
    bright = values > soft + (dense - soft) * level
    skeleton = largest_component(bone_mask(values, SPINE_LEVEL))       # позвоночник с тазом
    outside = bright & ~ndimage.binary_dilation(skeleton, np.ones((9, 9)))
    labels, count = ndimage.label(ndimage.binary_opening(outside, np.ones((2, 2))))
    if not count:
        return empty
    rows = values.shape[0]
    found = []
    for index in range(1, count + 1):
        ys, xs = np.nonzero(labels == index)
        size = len(ys)
        if size < 15 or size > values.size * max_area:
            continue
        box = max((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1), 1)
        found.append(dict(size=size, compactness=size / box, top=1.0 - ys.mean() / rows))
    if not found:
        return empty
    return dict(
        foreign_count=float(len(found)),
        foreign_area=float(sum(f["size"] for f in found) / values.size),
        foreign_max_area=float(max(f["size"] for f in found) / values.size),
        foreign_compactness=float(max(f["compactness"] for f in found)),
        foreign_top=float(max(f["top"] for f in found)),
    )


def spine_midline_angle(mask: np.ndarray, spacing: Spacing, signed: bool = False) -> tuple[float, float]:
    """Угол оси по центрам позвоночного столба в каждой строке — как меряет врач.

    Главная ось всей яркой области занижает наклон (её «тянет» вертикальная
    протяжённость), поэтому строим среднюю линию столба и считаем её наклон.
    """
    rows, cols = mask.shape
    band = mask[: int(rows * 0.8), int(cols * 0.2) : int(cols * 0.8)]
    centers, ys = [], []
    for row in range(band.shape[0]):
        xs = np.nonzero(band[row])[0]
        if len(xs) < 3:
            continue
        # столб — самая длинная связная группа в строке
        groups = np.split(xs, np.nonzero(np.diff(xs) > 1)[0] + 1)
        longest = max(groups, key=len)
        if len(longest) < 3:
            continue
        centers.append(float(longest.mean()))
        ys.append(row)
    if len(centers) < 20:
        return 0.0, 0.0
    mm_x, mm_y = spacing
    slope, _ = np.polyfit(np.array(ys, dtype=float), np.array(centers), 1)
    # знаковый угол нужен для поверки измерителя: поворот складывается с собственным наклоном
    angle = float(np.degrees(np.arctan2(slope * mm_x, mm_y)))
    if not signed:
        angle = abs(angle)
    residual = float(np.std(np.array(centers) - np.polyval([slope, np.mean(centers)], np.array(ys))))
    return angle, residual


def spine_features(values: np.ndarray, spacing: Spacing) -> dict:
    rows, cols = values.shape
    mm_x, mm_y = spacing
    mask = bone_mask(values, SPINE_LEVEL)
    midline_angle, midline_residual = spine_midline_angle(mask, spacing)
    # ось по всей яркой области — запасной признак, шкала занижена
    center = mask[: int(rows * 0.8), int(cols * 0.25) : int(cols * 0.75)]
    angle, ecc = _principal_axis(center, spacing)

    # охват: снизу должны быть видны гребни подвздошных костей, сверху — Th12 и рёбра
    bottom_corners = np.concatenate([mask[int(rows * 0.85) :, : int(cols * 0.2)].ravel(),
                                     mask[int(rows * 0.85) :, int(cols * 0.8) :].ravel()])
    top_corners = np.concatenate([mask[: int(rows * 0.2), : int(cols * 0.2)].ravel(),
                                  mask[: int(rows * 0.2), int(cols * 0.8) :].ravel()])
    # профиль костной плотности по строкам: сколько тел позвонков попало в кадр
    profile = mask[:, int(cols * 0.3) : int(cols * 0.7)].mean(axis=1)
    smooth = ndimage.uniform_filter1d(profile.astype(float), size=5)
    peaks = int(((smooth[1:-1] > smooth[:-2]) & (smooth[1:-1] >= smooth[2:]) & (smooth[1:-1] > 0.35)).sum())

    return dict(
        spine_axis_deg=angle,
        spine_midline_deg=midline_angle,
        spine_midline_over5=float(midline_angle > 5.0),
        spine_midline_residual=midline_residual,
        spine_axis_over5=float(angle > 5.0),
        spine_ecc=ecc,
        spine_height_cm=float(rows * mm_y / 10),
        spine_width_cm=float(cols * mm_x / 10),
        spine_iliac_signal=float(bottom_corners.mean()),
        spine_ribs_signal=float(top_corners.mean()),
        spine_top_cut=_edge_fraction(mask, "top"),
        spine_bottom_cut=_edge_fraction(mask, "bottom"),
        spine_margin_top_cm=_margin_cm(mask, "top", spacing),
        spine_margin_bottom_cm=_margin_cm(mask, "bottom", spacing),
        spine_vertebra_peaks=float(peaks),
        spine_bone_ratio=float(mask.mean()),
    )


def medial_side(mask: np.ndarray) -> str:
    """Медиальная сторона по анатомии: головка смещена медиально относительно диафиза.

    Надёжнее, чем вывод из метки «левое/правое бедро»: не зависит от эвристики стороны.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) < 50:
        return "left"
    y0, y1 = int(ys.min()), int(ys.max())
    height = max(y1 - y0, 1)
    top = xs[ys < y0 + height * 0.2]
    bottom = xs[ys > y1 - height * 0.25]
    if not len(top) or not len(bottom):
        return "left"
    return "right" if top.mean() > bottom.mean() else "left"


def rotation_geometry(mask: np.ndarray, spacing: Spacing) -> dict:
    """Измерения, по которым в ТЗ оценивают ротацию бедра.

    Малый вертел — локальная выпуклость медиального контура НИЖЕ шейки. Отклонение
    считается от хорды между началом и концом участка: так убирается общий изгиб кости
    и остаётся собственно выступ.
    """
    empty = dict(femur_troch_peak_mm=0.0, femur_troch_area_mm2=0.0, femur_troch_position=0.0,
                 femur_neck_width_mm=0.0, femur_head_diameter_mm=0.0, femur_neck_to_head=0.0,
                 femur_shaft_width_mm=0.0)
    ys, xs = np.nonzero(mask)
    if len(ys) < 80:
        return empty
    sx, sy = spacing
    y0, y1 = int(ys.min()), int(ys.max())
    height = max(y1 - y0, 1)
    side = medial_side(mask)

    def edge(row: int) -> float | None:
        cols = np.nonzero(mask[row])[0]
        return None if not len(cols) else float(cols.max() if side == "right" else cols.min())

    widths = np.array([mask[r].sum() for r in range(y0, y1 + 1)], dtype=float)
    upper = widths[: max(int(height * 0.6), 1)]
    neck_row = y0 + int(np.argmin(np.where(upper > 0, upper, np.inf)))
    neck_width = float(widths[neck_row - y0] * sx)

    head_diameter = 0.0
    if neck_row > y0 + 5:
        distance = ndimage.distance_transform_edt(mask[: neck_row + 1])
        if distance.size:
            head_diameter = float(2 * distance.max() * sx)

    shaft_rows = [r for r in range(int(y0 + height * 0.75), y1 + 1) if edge(r) is not None]
    if len(shaft_rows) < 8:
        return empty
    shaft_width = float(np.median([widths[r - y0] for r in shaft_rows]) * sx)

    band = [r for r in range(max(neck_row + 2, y0 + int(height * 0.3)), int(y0 + height * 0.85))
            if edge(r) is not None]
    if len(band) < 8:
        return empty
    rows_arr = np.array(band, dtype=float)
    contour = np.array([edge(r) for r in band], dtype=float)
    chord = np.interp(rows_arr, [rows_arr[0], rows_arr[-1]], [contour[0], contour[-1]])
    deviation = ((contour - chord) if side == "right" else (chord - contour)) * sx

    return dict(
        femur_troch_peak_mm=float(deviation.max()),
        femur_troch_area_mm2=float(np.clip(deviation, 0, None).sum() * sy),
        femur_troch_position=float((rows_arr[int(np.argmax(deviation))] - y0) / height),
        femur_neck_width_mm=neck_width,
        femur_head_diameter_mm=head_diameter,
        femur_neck_to_head=float(neck_width / head_diameter) if head_diameter > 0 else 0.0,
        femur_shaft_width_mm=shaft_width,
    )


def _shaft_and_trochanter(mask: np.ndarray, medial_side: str, spacing: Spacing) -> dict:
    """Малый вертел — выступ медиального контура над линией диафиза.

    По ТЗ ротацию оценивают именно по малому вертелу: при переротации контур
    гладкий, при недоротации вертел слишком крупный.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) < 50:
        return dict(femur_trochanter_bulge=0.0, femur_shaft_deg=0.0, femur_shaft_width_cm=0.0)
    y0, y1 = int(ys.min()), int(ys.max())
    height = y1 - y0

    def medial_edge(row: int) -> float | None:
        cols = np.nonzero(mask[row])[0]
        if not len(cols):
            return None
        return float(cols.max() if medial_side == "right" else cols.min())

    # диафиз — нижняя треть кости, там контур прямой
    shaft_rows = [r for r in range(int(y0 + height * 0.7), y1 + 1) if medial_edge(r) is not None]
    if len(shaft_rows) < 10:
        return dict(femur_trochanter_bulge=0.0, femur_shaft_deg=0.0, femur_shaft_width_cm=0.0)
    shaft_edges = np.array([medial_edge(r) for r in shaft_rows])
    slope, intercept = np.polyfit(np.array(shaft_rows, dtype=float), shaft_edges, 1)
    widths = np.array([mask[r].sum() for r in shaft_rows], dtype=float)

    # зона малого вертела — над диафизом
    bump_rows = [r for r in range(int(y0 + height * 0.38), int(y0 + height * 0.72))
                 if medial_edge(r) is not None]
    bulge = 0.0
    if bump_rows:
        actual = np.array([medial_edge(r) for r in bump_rows])
        expected = slope * np.array(bump_rows, dtype=float) + intercept
        deviation = (actual - expected) if medial_side == "right" else (expected - actual)
        bulge = float(np.clip(deviation, 0, None).mean() / max(np.median(widths), 1.0))

    mm_x, mm_y = spacing
    return dict(
        femur_trochanter_bulge=bulge,
        femur_shaft_deg=float(np.degrees(np.arctan2(abs(slope) * mm_x, mm_y))),
        femur_shaft_width_cm=float(np.median(widths) * mm_x / 10),
    )


def femur_features(values: np.ndarray, side: str | None, spacing: Spacing) -> dict:
    rows, cols = values.shape
    mm_x, mm_y = spacing
    all_bone = bone_mask(values, FEMUR_LEVEL)
    mask = largest_component(all_bone, prefer_bottom=True)  # только бедренная кость, без таза
    angle, ecc = _principal_axis(mask, spacing)

    # ТЗ: 3 см сверху и снизу, 2 см сбоку от области интереса
    margins = {s: _margin_cm(mask, s, spacing) for s in ("top", "bottom", "left", "right")}
    medial = "right" if side == "r" else "left"  # медиальная сторона зависит от бедра
    lateral = "left" if side == "r" else "right"

    # по ТЗ в кадре должны быть большой вертел, шейка и седалищная кость
    labels, count = ndimage.label(all_bone)
    sizes = np.bincount(labels.ravel())[1:] if count else np.array([])
    big = int((sizes > all_bone.size * 0.01).sum()) if count else 0
    pelvis_bone = all_bone & ~mask  # всё, что не бедренная кость: таз, седалищная кость
    pelvis = float(pelvis_bone[: int(rows * 0.35)].mean())
    ischium_band = pelvis_bone[int(rows * 0.3) : int(rows * 0.85)]
    ischium = float(ischium_band[:, int(cols * 0.6) :].mean() if medial == "right"
                    else ischium_band[:, : int(cols * 0.4)].mean())

    features = dict(
        femur_axis_deg=angle,
        femur_ecc=ecc,
        femur_height_cm=float(rows * mm_y / 10),
        femur_width_cm=float(cols * mm_x / 10),
        femur_margin_top_cm=margins["top"],
        femur_margin_bottom_cm=margins["bottom"],
        femur_margin_medial_cm=margins[medial],
        femur_margin_lateral_cm=margins[lateral],
        femur_margin_min_cm=float(min(margins.values())),
        femur_top_cut=_edge_fraction(mask, "top"),
        femur_bottom_cut=_edge_fraction(mask, "bottom"),
        femur_bone_ratio=float(mask.mean()),
        femur_components=float(big),
        femur_pelvis_signal=pelvis,
        femur_ischium_signal=ischium,
    )
    features.update(_shaft_and_trochanter(mask, medial, spacing))
    features.update(rotation_geometry(mask, spacing))
    # ротация плоха с обеих сторон: и переротация (вертел не виден), и недоротация
    # (вертел слишком крупный). Квадрат позволяет линейной модели поймать U-образную связь.
    features["femur_trochanter_sq"] = features["femur_trochanter_bulge"] ** 2
    features["femur_shaft_deg_sq"] = features["femur_shaft_deg"] ** 2
    return features


def compute(image: Image) -> dict:
    """Полный вектор признаков: геометрия + метаданные выгрузки."""
    values = normalize(image.array)
    spacing = getattr(image, "spacing", PIXEL_SPACING_MM)
    features = dict(
        rows=float(image.array.shape[0]),
        cols=float(image.array.shape[1]),
        spacing_x=float(spacing[0]),
        spacing_y=float(spacing[1]),
        copies=float(image.copies),          # найденный сигнал: переснятые исследования
        instance=float(image.instance),
        n_images=float(image.n_images),
        mean=float(values.mean()),
        std=float(values.std()),
        is_spine=float(image.region == "spine"),
    )
    features.update(foreign_body(values))
    if image.region == "spine":
        features.update(spine_features(values, spacing))
    else:
        features.update(femur_features(values, image.side, spacing))
    return features


SPINE_KEYS = ["spine_axis_deg", "spine_midline_deg", "spine_midline_over5", "spine_midline_residual",
              "spine_axis_over5", "spine_ecc", "spine_height_cm", "spine_width_cm",
              "spine_iliac_signal", "spine_ribs_signal", "spine_top_cut", "spine_bottom_cut",
              "spine_margin_top_cm", "spine_margin_bottom_cm", "spine_vertebra_peaks", "spine_bone_ratio"]
FEMUR_KEYS = ["femur_axis_deg", "femur_ecc", "femur_height_cm", "femur_width_cm",
              "femur_margin_top_cm", "femur_margin_bottom_cm", "femur_margin_medial_cm",
              "femur_margin_lateral_cm", "femur_margin_min_cm", "femur_top_cut", "femur_bottom_cut",
              "femur_bone_ratio", "femur_components", "femur_pelvis_signal", "femur_ischium_signal",
              "femur_trochanter_bulge", "femur_trochanter_sq", "femur_shaft_deg", "femur_shaft_deg_sq",
              "femur_shaft_width_cm", "femur_troch_peak_mm", "femur_troch_area_mm2",
              "femur_troch_position", "femur_neck_width_mm", "femur_head_diameter_mm",
              "femur_neck_to_head", "femur_shaft_width_mm"]
COMMON_KEYS = ["rows", "cols", "copies", "instance", "n_images", "mean", "std", "foreign_count",
               "foreign_area", "foreign_max_area", "foreign_compactness", "foreign_top"]

# Признаки под каждый критерий ТЗ: выбраны по смыслу критерия, а не подбором по данным.
CRITERION_FEATURES = {
    ("spine", "Не выравнена ось позвоночника"): ["spine_midline_deg", "spine_midline_over5",
                                                 "spine_axis_deg"],
    ("spine", "Некорректная укладка"): ["spine_iliac_signal", "spine_ribs_signal", "spine_bottom_cut",
                                        "spine_top_cut", "spine_margin_bottom_cm", "spine_vertebra_peaks"],
    ("spine", "Присутствуют посторонние предметы"): ["foreign_count", "foreign_area", "foreign_max_area",
                                                     "foreign_compactness", "foreign_top",
                                                     "spine_ribs_signal"],
    # Ротация по ТЗ оценивается по малому вертелу: при переротации контур гладкий.
    # Выступ описан двумя мерами (высота и площадь), ширина шейки ловит её укорочение.
    # Набор из трёх признаков: на 36 положительных примерах больше добавлять нельзя —
    # модель из шести признаков давала AUC 0.56 против 0.69 у этой тройки.
    ("femur", "Некорректная укладка"): ["femur_trochanter_bulge", "femur_troch_area_mm2",
                                        "femur_neck_width_mm"],
    ("femur", "Некорректная область интереса"): ["femur_margin_min_cm", "femur_margin_top_cm",
                                                 "femur_margin_bottom_cm", "femur_margin_medial_cm",
                                                 "femur_margin_lateral_cm", "femur_height_cm"],
}
