# -*- coding: utf-8 -*-
"""Итоговое сравнение наборов признаков на ЧЕСТНОЙ схеме с исправленным масштабом.

Масштаб пикселя берётся из Exposed Area, но с отсевом мусорной константы [520,478]
(она давала масштаб 1.7–1.9 мм вместо настоящих 0.60–0.65 и утекала метку).

    python scripts/final_compare.py --root "...\\_dsroot"
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dxa_real import evaluate as ev            # noqa: E402
from dxa_real import features as F             # noqa: E402
from dxa_real.data import (LABEL_AXIS, LABEL_FOREIGN, LABEL_POSITIONING,  # noqa: E402
                           LABEL_ROI, load_dataset)

# исходный «широкий» набор (как было в проекте)
V0 = {
    ("spine", LABEL_AXIS): ["spine_midline_deg", "spine_midline_over5", "spine_axis_deg"],
    ("spine", LABEL_POSITIONING): ["spine_iliac_signal", "spine_ribs_signal", "spine_bottom_cut",
                                   "spine_top_cut", "spine_margin_bottom_cm", "spine_vertebra_peaks"],
    ("spine", LABEL_FOREIGN): ["foreign_count", "foreign_area", "foreign_max_area",
                               "foreign_compactness", "foreign_top", "spine_ribs_signal"],
    ("femur", LABEL_POSITIONING): ["femur_trochanter_bulge", "femur_troch_area_mm2",
                                   "femur_neck_width_mm"],
    ("femur", LABEL_ROI): ["femur_margin_min_cm", "femur_margin_top_cm", "femur_margin_bottom_cm",
                           "femur_margin_medial_cm", "femur_margin_lateral_cm", "femur_height_cm"],
}

# текущий минимальный набор
V3 = {
    ("spine", LABEL_AXIS): ["spine_midline_deg"],
    ("spine", LABEL_POSITIONING): ["spine_iliac_signal", "spine_bottom_cut"],
    ("spine", LABEL_FOREIGN): ["spine_ribs_signal", "spine_vertebra_peaks"],
    ("femur", LABEL_POSITIONING): ["femur_trochanter_bulge", "femur_troch_area_mm2",
                                   "femur_neck_width_mm"],
    ("femur", LABEL_ROI): ["femur_height_cm", "femur_bone_ratio", "femur_margin_min_cm"],
}

# V3 + метаданные выгрузки (число копий снимка) — легальный сигнал вместо утечки
V3M = copy.deepcopy(V3)
V3M[("femur", LABEL_POSITIONING)] = V3[("femur", LABEL_POSITIONING)] + ["copies"]
V3M[("femur", LABEL_ROI)] = V3[("femur", LABEL_ROI)] + ["copies"]
V3M[("spine", LABEL_POSITIONING)] = V3[("spine", LABEL_POSITIONING)] + ["copies"]

# V4: набор L для укладки бедра (отступы поля сканирования + выступ вертела).
# Победитель честного перебора tune_femur_positioning: AUC укладки 0.552 -> 0.625.
V4 = copy.deepcopy(V3)
V4[("femur", LABEL_POSITIONING)] = ["femur_margin_min_cm", "femur_margin_top_cm",
                                    "femur_trochanter_bulge"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    images = load_dataset(args.root)
    table = ev.build_table(images)
    print(f"[data] изображений: {len(images)}")

    rows = []
    for name, fmap, mode in (("V0 (широкий)", V0, "nested"),
                             ("V3 (минимальный)", V3, "nested"),
                             ("V3 + copies", V3M, "nested"),
                             ("V3 (prior-порог)", V3, "prior"),
                             ("V4 (L-бедро, nested)", V4, "nested"),
                             ("V4 (L-бедро, prior)", V4, "prior")):
        F.CRITERION_FEATURES = copy.deepcopy(fmap)
        t0 = time.time()
        res = ev.evaluate(images, table, verbose=False, threshold_mode=mode)
        per = {k: s for k, s in res["scores"].items() if "ИТОГ" not in k}
        rows.append((name, res["macro_f1"], per))
        print(f"  {name:20} macroF1={res['macro_f1']:.3f}  ({time.time()-t0:.0f}s)")

    keys = sorted(rows[0][2].keys())
    short = [k.replace("spine: ", "S:").replace("femur: ", "F:")[:16] for k in keys]
    print("\n" + "=" * 120)
    print(f"{'набор':20} {'macroF1':>8} " + " ".join(f"{s:>16}" for s in short))
    for name, m, per in rows:
        print(f"{name:20} {m:8.3f} " + " ".join(f"{per[k].auc:8.3f}/{per[k].f1:.3f}" for k in keys))


if __name__ == "__main__":
    main()
