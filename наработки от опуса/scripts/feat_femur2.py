# -*- coding: utf-8 -*-
"""P2: признаки ротации бедра на ЧИСТОЙ маске (отделение таза от бедра).

Проблема (см. rotation.md и diag_femur): маска бедра сливается с тазом, шейка
меряется 40-50 мм вместо 30-35, контур вертела зашумлён — отсюда ложные
срабатывания femur_positioning (precision 0.37).

Решение Astra: морфологически разорвать тонкую перемычку головка-вертлужная
впадина (эрозия), взять компонент, доходящий до низа кадра (диафиз), и
восстановить его внутри исходной маски (реконструкция). Затем мерить ротацию и
нормировать на диаметр головки (конституционная инвариантность).

Запуск:
    python dxa_qc_work/scripts/feat_femur2.py
Результат: dxa_qc_work/out/feat_femur2.csv
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import _common  # noqa: F401

from src import config as C        # noqa: E402
from src import features as fl     # noqa: E402

FEMUR2_COLS = [
    "f2_neck_width_mm", "f2_head_diameter_mm", "f2_neck_to_head",
    "f2_troch_bulge", "f2_troch_area", "f2_troch_area_norm", "f2_troch_peak_norm",
    "f2_shaft_deg", "f2_axis_angle", "f2_bone_ratio", "f2_height_cm",
    "f2_margin_min_cm", "f2_margin_top_cm", "f2_margin_bottom_cm",
    "f2_head_to_width", "f2_components", "f2_pelvis_ratio",
]


def _empty() -> dict:
    return {c: 0.0 for c in FEMUR2_COLS}


def separate_femur(values: np.ndarray, level: float = fl.FEMUR_LEVEL,
                   erode: int = 5) -> tuple[np.ndarray, np.ndarray, int]:
    """Вернуть (маска бедра без таза, полная костная маска, число компонент).

    Тонкая перемычка головка-впадина рвётся эрозией; компонент, доходящий до низа
    кадра, — это диафиз бедра. Восстанавливаем его внутри полной маски
    (геодезическая реконструкция), чтобы вернуть настоящий контур.
    """
    from scipy import ndimage

    full = fl.bone_mask(values, level)
    labels, count = ndimage.label(full)
    if count <= 1:
        return full, full, count
    # разрыв тонких перемычек
    eroded = ndimage.binary_erosion(full, np.ones((erode, erode)))
    elab, ecount = ndimage.label(eroded)
    if ecount == 0:
        return full, full, count
    bottom = set(np.unique(elab[-3:])) - {0}
    if not bottom:
        # бедро может не доходить до низа при кропе — берём самую крупную
        sizes = np.bincount(elab.ravel())
        sizes[0] = 0
        seed = elab == int(np.argmax(sizes))
    else:
        sizes = np.bincount(elab.ravel())
        seed = elab == max(bottom, key=lambda i: sizes[i])
    # геодезическая реконструкция seed внутри full
    femur = seed.copy()
    while True:
        grown = ndimage.binary_dilation(femur, np.ones((3, 3))) & full
        if np.array_equal(grown, femur):
            break
        femur = grown
    return femur, full, count


def femur2_features(arr: np.ndarray, spacing=C.PIXEL_SPACING_MM) -> dict:
    from scipy import ndimage

    f = fl._as01(arr)
    rows, cols = f.shape
    mm_x, mm_y = spacing
    mask, full, ncomp = separate_femur(f)
    out = _empty()
    out["f2_components"] = float(ncomp)
    out["f2_pelvis_ratio"] = float(max(full.mean() - mask.mean(), 0.0) / max(full.mean(), 1e-9))
    ys, xs = np.nonzero(mask)
    if len(ys) < 80:
        return out

    angle, _ = fl._principal_axis(mask, spacing)
    out["f2_axis_angle"] = angle
    out["f2_bone_ratio"] = float(mask.mean())
    out["f2_height_cm"] = float(rows * mm_y / 10)
    for s, key in (("top", "f2_margin_top_cm"), ("bottom", "f2_margin_bottom_cm")):
        out[key] = fl._margin_cm(mask, s, spacing)
    out["f2_margin_min_cm"] = float(min(fl._margin_cm(mask, s, spacing)
                                       for s in ("top", "bottom", "left", "right")))

    y0, y1 = int(ys.min()), int(ys.max())
    height = max(y1 - y0, 1)
    side = fl.medial_side(mask)

    def edge(row: int):
        cc = np.nonzero(mask[row])[0]
        return None if not len(cc) else float(cc.max() if side == "right" else cc.min())

    widths = np.array([mask[r].sum() for r in range(y0, y1 + 1)], dtype=float)
    upper = widths[: max(int(height * 0.6), 1)]
    neck_row = y0 + int(np.argmin(np.where(upper > 0, upper, np.inf)))
    neck_width = float(widths[neck_row - y0] * mm_x)
    out["f2_neck_width_mm"] = neck_width

    head_diameter = 0.0
    if neck_row > y0 + 5:
        dist = ndimage.distance_transform_edt(mask[: neck_row + 1])
        if dist.size:
            head_diameter = float(2 * dist.max() * mm_x)
    out["f2_head_diameter_mm"] = head_diameter
    out["f2_neck_to_head"] = float(neck_width / head_diameter) if head_diameter > 0 else 0.0
    out["f2_head_to_width"] = float(head_diameter / (cols * mm_x)) if cols else 0.0

    band = [r for r in range(max(neck_row + 2, y0 + int(height * 0.3)), int(y0 + height * 0.85))
            if edge(r) is not None]
    bulge = area = peak = 0.0
    if len(band) >= 8:
        rows_arr = np.array(band, dtype=float)
        contour = np.array([edge(r) for r in band], dtype=float)
        chord = np.interp(rows_arr, [rows_arr[0], rows_arr[-1]], [contour[0], contour[-1]])
        dev = ((contour - chord) if side == "right" else (chord - contour)) * mm_x
        bulge = float(dev.max())
        area = float(np.clip(dev, 0, None).sum() * mm_y)
        peak = float(np.clip(dev, 0, None).max())
    out["f2_troch_bulge"] = bulge
    out["f2_troch_area"] = area
    # нормировка на диаметр головки (конституционная инвариантность).
    # Диаметр может не определиться (0) — тогда нормировка не считается, иначе деление
    # на ~0 даёт выбросы в 1e15 и ломает линейную модель.
    if head_diameter > 5.0:
        out["f2_troch_area_norm"] = float(area / (head_diameter ** 2))
        out["f2_troch_peak_norm"] = float(peak / head_diameter)
    else:
        out["f2_troch_area_norm"] = 0.0
        out["f2_troch_peak_norm"] = 0.0

    shaft_rows = [r for r in range(int(y0 + height * 0.75), y1 + 1) if edge(r) is not None]
    if len(shaft_rows) >= 8:
        edges = np.array([edge(r) for r in shaft_rows])
        slope, _ = np.polyfit(np.array(shaft_rows, dtype=float), edges, 1)
        out["f2_shaft_deg"] = float(np.degrees(np.arctan2(abs(slope) * mm_x, mm_y)))
    return out


def main():
    out_csv = os.path.join(_common.OUT, "feat_femur2.csv")
    df = pd.read_csv(C.MANIFEST_CSV)
    rows = []
    for i, r in df.iterrows():
        try:
            arr = fl._load_for_features(r)
            sp = (float(r["spacing_x"]), float(r["spacing_y"])) \
                if pd.notna(r.get("spacing_x")) else C.PIXEL_SPACING_MM
            feat = femur2_features(arr, sp)
        except Exception as exc:  # noqa: BLE001
            print(f"  ошибка на {r['image_uid']}: {exc}")
            feat = _empty()
        feat["image_uid"] = r["image_uid"]
        rows.append(feat)
        if (i + 1) % 50 == 0:
            print(f"  обработано {i + 1}/{len(df)}")
    out = pd.DataFrame(rows).drop_duplicates("image_uid", keep="first")
    out.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\nсохранено: {out_csv}  строк: {len(out)}")
    print(out[FEMUR2_COLS].describe().to_string())


if __name__ == "__main__":
    main()
