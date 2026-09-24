# -*- coding: utf-8 -*-
"""Эксперимент: какая оценка наклона оси позвоночника лучше отделяет метку.

Проблема: текущий ``spine_midline_angle`` даёт AUC 0.827, но F1@prior всего 0.300 —
при k=10 в top попадает 3 истинных. Причина — сильное перекрытие: у 7 из 10
нарушений измеренный угол < 6°, тогда как у нормы встречается до 11.5°. То есть
само измерение (устойчивость подгонки средней линии) — узкое место.

Здесь считаем несколько вариантов оценки угла по одной и той же костной маске
и сравниваем AUC / AP / F1@prior по метке ``spine_axis``.

Запуск:
    cd dxa_qc && python ../scripts/exp_axis_measure.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dxa_qc"))

from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import dicom_io           # noqa: E402
from src import features as fl     # noqa: E402
from src.train import prior_threshold  # noqa: E402


def _runs(band_row):
    xs = np.nonzero(band_row)[0]
    if len(xs) < 3:
        return []
    return np.split(xs, np.nonzero(np.diff(xs) > 1)[0] + 1)


def _row_centers(mask, col_lo=0.2, col_hi=0.8, row_hi=0.8, min_len=3,
                 min_rel=0.0):
    """Центры самой длинной связной группы кости в каждой строке полосы."""
    rows, cols = mask.shape
    band = mask[: int(rows * row_hi), int(cols * col_lo): int(cols * col_hi)]
    out = []
    for row in range(band.shape[0]):
        groups = _runs(band[row])
        if not groups:
            continue
        longest = max(groups, key=len)
        if len(longest) < min_len:
            continue
        out.append((float(row), float(longest.mean()), float(len(longest))))
    if not out:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    a = np.array(out)
    if min_rel > 0 and len(a):
        keep = a[:, 2] >= min_rel * a[:, 2].max()
        a = a[keep]
    return a[:, 0], a[:, 1], a[:, 2]


def _angle_from_slope(slope, spacing):
    sx, sy = spacing
    return float(np.degrees(np.arctan2(abs(slope) * sx, sy)))


def m_current(mask, spacing):
    ys, cx, _ = _row_centers(mask)
    if len(cx) < 20:
        return 0.0
    slope, _ = np.polyfit(ys, cx, 1)
    return _angle_from_slope(slope, spacing)


def m_wide(mask, spacing):
    """Только «широкие» строки (тела позвонков), отсев тонких (рёбра/отростки)."""
    ys, cx, _ = _row_centers(mask, min_rel=0.5)
    if len(cx) < 20:
        return 0.0
    slope, _ = np.polyfit(ys, cx, 1)
    return _angle_from_slope(slope, spacing)


def m_theil(mask, spacing):
    """Робастный наклон (Theil-Sen) — устойчив к выбросным строкам."""
    from scipy.stats import theilslopes
    ys, cx, _ = _row_centers(mask)
    if len(cx) < 20:
        return 0.0
    slope = theilslopes(cx, ys)[0]
    return _angle_from_slope(slope, spacing)


def m_weighted(mask, spacing):
    """Взвешенная подгонка: вес строки — длина костной группы (тело позвонка)."""
    ys, cx, w = _row_centers(mask)
    if len(cx) < 20:
        return 0.0
    slope = np.polyfit(ys, cx, 1, w=w)[0]
    return _angle_from_slope(slope, spacing)


def m_center_diff(mask, spacing):
    """Наклон по разнице медианных центров верхней и нижней четвертей."""
    ys, cx, _ = _row_centers(mask)
    if len(cx) < 20:
        return 0.0
    lo, hi = np.percentile(ys, 25), np.percentile(ys, 75)
    top = cx[ys <= lo]
    bot = cx[ys >= hi]
    if len(top) < 5 or len(bot) < 5:
        return 0.0
    dy = (np.median(ys[ys >= hi]) - np.median(ys[ys <= lo]))
    if dy <= 0:
        return 0.0
    slope = (np.median(bot) - np.median(top)) / dy
    return _angle_from_slope(slope, spacing)


def m_central(mask, spacing):
    """Узкая центральная полоса (0.3-0.7) — меньше шансов поймать ребро."""
    ys, cx, _ = _row_centers(mask, col_lo=0.3, col_hi=0.7)
    if len(cx) < 20:
        return 0.0
    slope, _ = np.polyfit(ys, cx, 1)
    return _angle_from_slope(slope, spacing)


def m_largest(mask, spacing):
    """Сначала оставляем только крупнейшую связную кость (сам столб)."""
    m = fl.largest_component(mask)
    ys, cx, _ = _row_centers(m)
    if len(cx) < 20:
        return 0.0
    slope, _ = np.polyfit(ys, cx, 1)
    return _angle_from_slope(slope, spacing)


METHODS = {
    "текущий": m_current,
    "широкие_строки": m_wide,
    "theil_sen": m_theil,
    "взвеш_шириной": m_weighted,
    "разница_центров": m_center_diff,
    "узкая_полоса": m_central,
    "крупнейшая_кость": m_largest,
}


def main() -> None:
    manifest = ds.load_manifest(C.MANIFEST_CSV)
    real = np.asarray(ds.real_mask(manifest))
    sp = manifest[(manifest["region"] == C.REGION_SPINE).values & real].copy()
    y = sp["viol_spine_axis"].values.astype(float)
    known = ~np.isnan(y)
    sp = sp[known]
    y = y[known].astype(int)
    print(f"снимков позвоночника: {len(sp)}, нарушений оси: {int(y.sum())}\n")

    vals = {k: [] for k in METHODS}
    for n, (_, r) in enumerate(sp.iterrows(), 1):
        ds_dcm, arr = dicom_io.read_dicom(r["source_path"])
        arr = np.asarray(arr)
        spacing = dicom_io.pixel_spacing(ds_dcm, arr) or C.PIXEL_SPACING_MM
        mask = fl.bone_mask(fl._as01(arr), fl.SPINE_LEVEL)
        for k, fn in METHODS.items():
            try:
                vals[k].append(fn(mask, spacing))
            except Exception:
                vals[k].append(0.0)
        if n % 25 == 0:
            print(f"  обработано {n}/{len(sp)}")

    print(f"\n{'метод':20} {'AUC':>6} {'AP':>6} {'F1@prior':>9} {'TP/k':>7}")
    rows = []
    for k in METHODS:
        v = np.array(vals[k], dtype=float)
        auc = roc_auc_score(y, v)
        ap = average_precision_score(y, v)
        thr = prior_threshold(y, v)
        pred = (v >= thr).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        rows.append((f1, auc, ap, k, int(pred.sum()), int(((y == 1) & (pred == 1)).sum())))
    for f1, auc, ap, k, kk, tp in sorted(rows, reverse=True):
        print(f"{k:20} {auc:6.3f} {ap:6.3f} {f1:9.3f} {tp:3d}/{kk:<3d}")

    print("\nуглы по классам для лучшего метода:")
    best = max(rows)[3]
    v = np.array(vals[best], dtype=float)
    for cls, lab in ((0, "норма"), (1, "нарушение")):
        vv = v[y == cls]
        print(f"  {lab:10} n={len(vv):3d} медиана={np.median(vv):5.2f} "
              f"q25={np.percentile(vv,25):5.2f} q75={np.percentile(vv,75):5.2f}")


if __name__ == "__main__":
    main()
