# -*- coding: utf-8 -*-
"""Геометрические признаки контроля качества DXA (гибридный подход).

Идея: критерии ТЗ — это измеримые величины, поэтому их надо мерить, а не
угадывать сетью. Признаки перенесены из честного измерительного контура
(`dxa_real/features.py`) и подобраны под каждый критерий по результатам
кросс-валидации (см. CRITERION_FEATURES).

Позвоночник:
  * ось — угол средней линии столба (spine_midline_angle), допуск 5°;
  * укладка — видимость гребней подвздошных костей снизу и Th12/рёбер сверху;
  * посторонние предметы — яркие объекты вне костных структур.
Бедро:
  * область интереса — отступы поля сканирования (3 см сверху/снизу, 2 см сбоку);
  * ротация/укладка — малый вертел как выпуклость медиального контура ниже шейки
    и ширина шейки.

Все линейные и угловые величины пересчитываются в физические единицы с учётом
реального размера пикселя снимка (spacing из Exposed Area, см. dicom_io).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import config as C

FEATURES_CSV = os.path.join(C.ARTIFACTS_DIR, "features.csv")

# Полный список признаков (имена фиксированы, порядок важен для отчёта).
FEATURE_NAMES = [
    # позвоночник
    "spine_axis_angle", "spine_midline_angle", "spine_midline_residual",
    "spine_iliac_signal", "spine_ribs_signal", "spine_vertebra_peaks",
    "spine_top_cut", "spine_bottom_cut", "spine_margin_bottom_cm",
    "spine_height_cm", "spine_bone_ratio",
    # бедро
    "femur_axis_angle", "femur_bone_ratio", "femur_height_cm", "femur_width_cm",
    "femur_margin_min_cm", "femur_margin_top_cm", "femur_margin_bottom_cm",
    "femur_trochanter_bulge", "femur_troch_area_mm2", "femur_neck_width_mm",
    "femur_head_diameter_mm", "femur_neck_to_head", "femur_shaft_deg",
    # общие
    "bone_eccentricity", "bright_area_ratio", "vertical_symmetry", "copies",
]

# --- базовые примитивы ------------------------------------------------------ #

SPINE_LEVEL = 0.25   # контур позвонков и гребней без мягких тканей
FEMUR_LEVEL = 0.15   # вся бедренная кость, включая губчатую часть шейки


def _as01(arr: np.ndarray) -> np.ndarray:
    """Привести изображение (uint8 или float) к float [0,1]."""
    f = np.asarray(arr).astype(np.float32)
    if f.max() > 1.0:
        f = f / 255.0
    return np.clip(f, 0.0, 1.0)


def bone_mask(values: np.ndarray, level: float = 0.2) -> np.ndarray:
    """Кость целиком (порог между мягкими тканями и плотной костью + заливка дыр).

    Порог Оцу на DXA отсекает губчатую часть и даёт рваный контур, поэтому берём
    долю между медианой мягких тканей и плотной костью.
    """
    from scipy import ndimage

    soft, dense = np.percentile(values, (50, 98))
    if dense <= soft:
        return np.zeros_like(values, dtype=bool)
    mask = values > soft + (dense - soft) * level
    mask = ndimage.binary_closing(mask, np.ones((5, 5)))
    mask = ndimage.binary_fill_holes(mask)
    return ndimage.binary_opening(mask, np.ones((3, 3)))


def largest_component(mask: np.ndarray, prefer_bottom: bool = False) -> np.ndarray:
    """Самая крупная связная кость; с prefer_bottom — та, что доходит до низа кадра
    (для бедра это диафиз, что отделяет бедренную кость от таза)."""
    from scipy import ndimage

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


def _principal_axis(mask: np.ndarray, spacing=(1.0, 1.0)):
    """Главная ось бинарной маски: (угол к вертикали в град., эксцентриситет)."""
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
    angle = np.degrees(np.arctan2(abs(main[0]), abs(main[1]) + 1e-9))
    ecc = float(np.sqrt(max(vals.max(), 1e-9) / max(vals.min(), 1e-9)))
    return float(angle), ecc


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


def _margin_cm(mask: np.ndarray, side: str, spacing) -> float:
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


# --- позвоночник ------------------------------------------------------------ #

def spine_axis_angle(arr: np.ndarray, thr_pct: float = 90.0,
                     spacing=C.PIXEL_SPACING_MM) -> float:
    """Запасной признак: главная ось яркой области (шкала занижена)."""
    f = _as01(arr)
    h, w = f.shape
    roi = f[:, int(w * 0.15):int(w * 0.85)]
    if roi.max() <= roi.min():
        return 0.0
    mask = roi >= np.percentile(roi, thr_pct)
    angle, _ = _principal_axis(mask, spacing)
    return angle


def _midline_from_mask(mask: np.ndarray, spacing):
    """Наклон средней линии столба по бинарной костной маске (как меряет врач).

    Возвращает (угол к вертикали в град., остаточный разброс). Считается по
    центрам самой длинной связной группы в каждой строке.
    """
    rows, cols = mask.shape
    band = mask[: int(rows * 0.8), int(cols * 0.2): int(cols * 0.8)]
    centers, ys = [], []
    for row in range(band.shape[0]):
        xs = np.nonzero(band[row])[0]
        if len(xs) < 3:
            continue
        groups = np.split(xs, np.nonzero(np.diff(xs) > 1)[0] + 1)
        longest = max(groups, key=len)
        if len(longest) < 3:
            continue
        centers.append(float(longest.mean()))
        ys.append(float(row))
    if len(centers) < 20:
        return 0.0, 0.0
    ys_arr, cx = np.array(ys), np.array(centers)
    slope, _ = np.polyfit(ys_arr, cx, 1)
    sx, sy = spacing
    angle = float(np.degrees(np.arctan2(abs(slope) * sx, sy)))
    residual = float(np.std(cx - np.polyval([slope, cx.mean()], ys_arr)))
    return angle, residual


def spine_midline_angle(arr: np.ndarray, spacing=C.PIXEL_SPACING_MM,
                        return_residual: bool = False):
    """Наклон оси по средней линии позвоночного столба (градусы к вертикали).

    Как меряет врач: линия через центры тел позвонков. На наборе организатора этот
    признак отделяет метку «ось не выровнена» заметно лучше главной оси всей яркой
    области (AUC 0.83 против 0.78), а значения совпадают со шкалой ТЗ (медиана ~5°
    при нарушении против ~2.5° в норме).
    """
    mask = bone_mask(_as01(arr), SPINE_LEVEL)
    return _midline_from_mask(mask, spacing) if return_residual \
        else _midline_from_mask(mask, spacing)[0]


def spine_features(arr: np.ndarray, spacing=C.PIXEL_SPACING_MM) -> dict:
    """Полный набор признаков позвоночника (охват кадра, ось, кость)."""
    from scipy import ndimage

    f = _as01(arr)
    rows, cols = f.shape
    mm_x, mm_y = spacing
    mask = bone_mask(f, SPINE_LEVEL)
    # средняя линия — по костной маске (совпадает с честным контуром dxa_real)
    midline, residual = _midline_from_mask(mask, spacing)
    center = mask[: int(rows * 0.8), int(cols * 0.25): int(cols * 0.75)]
    angle, _ecc = _principal_axis(center, spacing)

    # охват: снизу видны гребни подвздошных костей, сверху — Th12 и рёбра
    bottom_corners = np.concatenate([mask[int(rows * 0.85):, : int(cols * 0.2)].ravel(),
                                     mask[int(rows * 0.85):, int(cols * 0.8):].ravel()])
    top_corners = np.concatenate([mask[: int(rows * 0.2), : int(cols * 0.2)].ravel(),
                                  mask[: int(rows * 0.2), int(cols * 0.8):].ravel()])
    profile = mask[:, int(cols * 0.3): int(cols * 0.7)].mean(axis=1)
    smooth = ndimage.uniform_filter1d(profile.astype(float), size=5)
    peaks = int(((smooth[1:-1] > smooth[:-2]) & (smooth[1:-1] >= smooth[2:])
                 & (smooth[1:-1] > 0.35)).sum())
    return {
        "spine_axis_angle": angle,
        "spine_midline_angle": midline,
        "spine_midline_residual": residual,
        "spine_iliac_signal": float(bottom_corners.mean()),
        "spine_ribs_signal": float(top_corners.mean()),
        "spine_vertebra_peaks": float(peaks),
        "spine_top_cut": _edge_fraction(mask, "top"),
        "spine_bottom_cut": _edge_fraction(mask, "bottom"),
        "spine_margin_bottom_cm": _margin_cm(mask, "bottom", spacing),
        "spine_height_cm": float(rows * mm_y / 10),
        "spine_bone_ratio": float(mask.mean()),
    }


# --- бедро ------------------------------------------------------------------ #

def femur_axis_angle(arr: np.ndarray, thr_pct: float = 92.0,
                     spacing=C.PIXEL_SPACING_MM) -> float:
    """Наклон главной оси бедренной кости (в градусах от вертикали)."""
    f = _as01(arr)
    if f.max() <= f.min():
        return 0.0
    mask = f >= np.percentile(f, thr_pct)
    angle, _ = _principal_axis(mask, spacing)
    return angle


def medial_side(mask: np.ndarray) -> str:
    """Медиальная сторона по анатомии: головка смещена медиально относительно диафиза."""
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


def rotation_geometry(mask: np.ndarray, spacing) -> dict:
    """Малый вертел (выпуклость ниже шейки), ширина шейки и головки.

    Отклонение считается от хорды между началом и концом участка — так убирается
    общий изгиб кости и остаётся собственно выступ. Измерение поверено поворотом и
    обрезкой реальных снимков (см. docs/validation.md).
    """
    from scipy import ndimage

    empty = dict(femur_trochanter_bulge=0.0, femur_troch_area_mm2=0.0,
                 femur_neck_width_mm=0.0, femur_head_diameter_mm=0.0,
                 femur_neck_to_head=0.0, femur_shaft_deg=0.0)
    ys, xs = np.nonzero(mask)
    if len(ys) < 80:
        return empty
    sx, sy = spacing
    y0, y1 = int(ys.min()), int(ys.max())
    height = max(y1 - y0, 1)
    side = medial_side(mask)

    def edge(row: int):
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

    # малый вертел — выпуклость медиального контура на участке ниже шейки
    band = [r for r in range(max(neck_row + 2, y0 + int(height * 0.3)), int(y0 + height * 0.85))
            if edge(r) is not None]
    bulge = area = 0.0
    if len(band) >= 8:
        rows_arr = np.array(band, dtype=float)
        contour = np.array([edge(r) for r in band], dtype=float)
        chord = np.interp(rows_arr, [rows_arr[0], rows_arr[-1]], [contour[0], contour[-1]])
        dev = ((contour - chord) if side == "right" else (chord - contour)) * sx
        bulge = float(dev.max())
        area = float(np.clip(dev, 0, None).sum() * sy)

    # наклон диафиза (нижняя треть)
    shaft_rows = [r for r in range(int(y0 + height * 0.75), y1 + 1) if edge(r) is not None]
    shaft_deg = 0.0
    if len(shaft_rows) >= 8:
        edges = np.array([edge(r) for r in shaft_rows])
        slope, _ = np.polyfit(np.array(shaft_rows, dtype=float), edges, 1)
        shaft_deg = float(np.degrees(np.arctan2(abs(slope) * sx, sy)))

    return dict(
        femur_trochanter_bulge=bulge,
        femur_troch_area_mm2=area,
        femur_neck_width_mm=neck_width,
        femur_head_diameter_mm=head_diameter,
        femur_neck_to_head=float(neck_width / head_diameter) if head_diameter > 0 else 0.0,
        femur_shaft_deg=shaft_deg,
    )


def _shaft_and_trochanter(mask: np.ndarray, medial_side: str, spacing) -> dict:
    """Малый вертел как выступ медиального контура над линией диафиза (нормированный).

    Дополняет rotation_geometry: там выступ измеряется в мм от хорды, здесь — как
    относительная выпуклость над продолжением диафиза, устойчивая к масштабу снимка.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) < 50:
        return dict(femur_trochanter_bulge=0.0, femur_shaft_deg=0.0, femur_shaft_width_cm=0.0)
    y0, y1 = int(ys.min()), int(ys.max())
    height = y1 - y0

    def medial_edge(row: int):
        cols = np.nonzero(mask[row])[0]
        if not len(cols):
            return None
        return float(cols.max() if medial_side == "right" else cols.min())

    shaft_rows = [r for r in range(int(y0 + height * 0.7), y1 + 1) if medial_edge(r) is not None]
    if len(shaft_rows) < 10:
        return dict(femur_trochanter_bulge=0.0, femur_shaft_deg=0.0, femur_shaft_width_cm=0.0)
    shaft_edges = np.array([medial_edge(r) for r in shaft_rows])
    slope, intercept = np.polyfit(np.array(shaft_rows, dtype=float), shaft_edges, 1)
    widths = np.array([mask[r].sum() for r in shaft_rows], dtype=float)

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


def femur_features(arr: np.ndarray, spacing=C.PIXEL_SPACING_MM) -> dict:
    """Полный набор признаков бедра (область интереса + ротация)."""
    f = _as01(arr)
    rows, cols = f.shape
    mm_x, mm_y = spacing
    all_bone = bone_mask(f, FEMUR_LEVEL)
    mask = largest_component(all_bone, prefer_bottom=True)
    angle, ecc = _principal_axis(mask, spacing)

    margins = {s: _margin_cm(mask, s, spacing) for s in ("top", "bottom", "left", "right")}
    feats = dict(
        femur_axis_angle=angle,
        femur_bone_ratio=float(mask.mean()),
        femur_height_cm=float(rows * mm_y / 10),
        femur_width_cm=float(cols * mm_x / 10),
        femur_margin_min_cm=float(min(margins.values())),
        femur_margin_top_cm=margins["top"],
        femur_margin_bottom_cm=margins["bottom"],
    )
    feats.update(rotation_geometry(mask, spacing))
    # нормированный выступ вертела/наклон диафиза перекрывают значения из rotation_geometry
    feats.update(_shaft_and_trochanter(mask, medial_side(mask), spacing))
    return feats


# --- общие ------------------------------------------------------------------ #

def bone_eccentricity(arr: np.ndarray, thr_pct: float = 92.0,
                      spacing=C.PIXEL_SPACING_MM) -> float:
    f = _as01(arr)
    if f.max() <= f.min():
        return 0.0
    _angle, ecc = _principal_axis(f >= np.percentile(f, thr_pct), spacing)
    return ecc


def bright_area_ratio(arr: np.ndarray, thr_pct: float = 92.0) -> float:
    f = _as01(arr)
    if f.max() <= f.min():
        return 0.0
    return float((f >= np.percentile(f, thr_pct)).mean())


def vertical_symmetry(arr: np.ndarray) -> float:
    f = _as01(arr)
    if f.max() <= f.min():
        return 0.0
    mask = (f >= np.percentile(f, 90)).astype(np.float32)
    col = mask.sum(0)
    if col.sum() <= 0:
        return 0.0
    left = col[: len(col) // 2].sum()
    right = col[len(col) // 2:].sum()
    return float(abs(left - right) / (col.sum() + 1e-9))


def compute_features(arr: np.ndarray, spacing=C.PIXEL_SPACING_MM,
                     region: str | None = None, copies: float = 1.0) -> dict:
    """Полный вектор геометрических признаков для изображения.

    region: если известна («lumbar_spine» / «proximal_femur_*»), считается только
    нужная группа признаков (остальные — 0), что ускоряет обработку. Если None —
    считаются обе группы.
    """
    sx, sy = spacing if spacing else C.PIXEL_SPACING_MM
    out = {
        "bone_eccentricity": bone_eccentricity(arr, spacing=(sx, sy)),
        "bright_area_ratio": bright_area_ratio(arr),
        "vertical_symmetry": vertical_symmetry(arr),
        "copies": float(copies),
    }
    is_spine = region is None or region == C.REGION_SPINE
    is_femur = region is None or region in (C.REGION_FEMUR_LEFT, C.REGION_FEMUR_RIGHT)
    if is_spine:
        out.update(spine_features(arr, spacing=(sx, sy)))
    if is_femur:
        out.update(femur_features(arr, spacing=(sx, sy)))
    for name in FEATURE_NAMES:
        out.setdefault(name, 0.0)
    return out


def _load_for_features(row, verbose: bool = False) -> np.ndarray:
    """Загрузить изображение для признаков в «сыром» виде.

    Признаки аффинно-инвариантны (все пороги — по перцентилям), но НЕ инвариантны
    к нелинейным преобразованиям. Кэш PNG хранит изображение после modality/VOI LUT
    (отсечение окном) — на нём признаки смещаются и перестают совпадать с теми, что
    видит инференс (он получает сырой массив). Поэтому читаем исходный DICOM.

    Для СИНТЕТИЧЕСКИХ строк source_path указывает на чистый исходник, а само
    нарушение лежит только в кэше, — поэтому такие строки читаем из кэша.
    """
    synthetic = bool(row.get("synthetic", False))
    src = row.get("source_path")
    if not synthetic and isinstance(src, str) and os.path.isfile(src):
        try:
            from . import dicom_io
            _ds, arr = dicom_io.read_dicom(src)
            if arr is not None and arr.size:
                return np.asarray(arr)
        except Exception:
            pass
    from PIL import Image
    return np.asarray(Image.open(row["cache_path"]).convert("L"))


def build_features(manifest_csv: str = C.MANIFEST_CSV,
                   out_csv: str = FEATURES_CSV, verbose: bool = True) -> pd.DataFrame:
    """Вычислить геометрические признаки для всех изображений манифеста."""
    df = pd.read_csv(manifest_csv)
    rows = []
    for _, r in df.iterrows():
        arr = _load_for_features(r)
        if "spacing_x" in df.columns and pd.notna(r.get("spacing_x")):
            spacing = (float(r["spacing_x"]), float(r["spacing_y"]))
        else:
            spacing = C.PIXEL_SPACING_MM
        copies = float(r["copies"]) if "copies" in df.columns and pd.notna(r.get("copies")) else 1.0
        feat = compute_features(arr, spacing, region=r.get("region"), copies=copies)
        feat["image_uid"] = r["image_uid"]
        feat["study_uid"] = r["study_uid"]
        feat["region"] = r.get("region")
        rows.append(feat)
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False, encoding="utf-8")
    if verbose:
        print(f"[features] сохранено: {out_csv}  строк: {len(out)}")
        cols = [c for c in FEATURE_NAMES if c in out.columns]
        print(out[cols].describe().to_string())
    return out


def load_features(manifest: pd.DataFrame,
                  features_csv: str = FEATURES_CSV) -> pd.DataFrame:
    """Присоединить геометрические признаки к манифесту (по image_uid)."""
    if not os.path.isfile(features_csv):
        build_features(out_csv=features_csv)
    feat = pd.read_csv(features_csv)
    # один image_uid = одна строка признаков: дубликаты в манифесте (одинаковый
    # источник + одинаковое синтетическое преобразование) не должны размножать
    # строки при merge и ломать выравнивание с OOF-массивами.
    feat = feat.drop_duplicates("image_uid", keep="first")
    keep = ["image_uid"] + [c for c in FEATURE_NAMES if c in feat.columns]
    return manifest.merge(feat[keep], on="image_uid", how="left")
