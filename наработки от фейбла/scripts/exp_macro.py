# -*- coding: utf-8 -*-
"""Влияние наборов признаков на ИТОГОВУЮ macro-F1 (честная схема dxa_real).

Читает кэш признаков (out/feat_cache.csv, включает dlr_*), подменяет
CRITERION_FEATURES и считает macro-F1 через dxa_real.evaluate.

Запуск:
    python scripts/exp_macro.py [--mode nested|prior|blend]
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dxa_real import evaluate as ev          # noqa: E402
from dxa_real import features as F           # noqa: E402
from dxa_real.data import (LABEL_AXIS, LABEL_FOREIGN, LABEL_POSITIONING,  # noqa: E402
                           LABEL_ROI)

CACHE = os.path.join("out", "feat_cache.csv")

V3 = {
    ("spine", LABEL_AXIS): ["spine_midline_deg"],
    ("spine", LABEL_POSITIONING): ["spine_iliac_signal", "spine_bottom_cut"],
    ("spine", LABEL_FOREIGN): ["spine_ribs_signal", "spine_vertebra_peaks"],
    ("femur", LABEL_POSITIONING): ["femur_trochanter_bulge", "femur_troch_area_mm2",
                                   "femur_neck_width_mm"],
    ("femur", LABEL_ROI): ["femur_height_cm", "femur_bone_ratio", "femur_margin_min_cm"],
}


def variant(femur_pos, femur_roi=None, spine_foreign=None, spine_pos=None, spine_axis=None):
    v = copy.deepcopy(V3)
    if femur_pos is not None:
        v[("femur", LABEL_POSITIONING)] = femur_pos
    if femur_roi is not None:
        v[("femur", LABEL_ROI)] = femur_roi
    if spine_foreign is not None:
        v[("spine", LABEL_FOREIGN)] = spine_foreign
    if spine_pos is not None:
        v[("spine", LABEL_POSITIONING)] = spine_pos
    if spine_axis is not None:
        v[("spine", LABEL_AXIS)] = spine_axis
    return v


VARIANTS = {
    "V3 baseline": V3,
    "femur=dlr_bone_ratio": variant(["dlr_femur_bone_ratio"]),
    "femur=dlr_bone_ratio+shaft_deg": variant(["dlr_femur_bone_ratio", "femur_shaft_deg"]),
    "femur=shaft_deg+troch_pos": variant(["femur_shaft_deg", "femur_troch_position"]),
    "femur=shaft_deg_sq+troch_pos+marg_lat": variant(["femur_shaft_deg_sq", "femur_troch_position",
                                                      "femur_margin_lateral_cm"]),
    "artifacts=+residual": variant(None, spine_foreign=["spine_ribs_signal",
                                                        "spine_vertebra_peaks",
                                                        "spine_midline_residual"]),
    "BEST combo (femur+artifacts)": variant(["dlr_femur_bone_ratio", "femur_shaft_deg"],
                                            spine_foreign=["spine_ribs_signal",
                                                           "spine_vertebra_peaks",
                                                           "spine_midline_residual"]),
    "BEST2 (femur univ + artifacts+res)": variant(["dlr_femur_bone_ratio"],
                                                  spine_foreign=["spine_ribs_signal",
                                                                 "spine_vertebra_peaks",
                                                                 "spine_midline_residual"]),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="nested", choices=["nested", "prior", "blend"])
    args = ap.parse_args()

    table = pd.read_csv(CACHE)
    print(f"[macro] таблица: {len(table)} строк, режим={args.mode}\n")
    print(f"{'вариант':40} {'macroF1':>8} {'spineQ':>7} {'femurQ':>7} "
          f"{'femurPos F1':>11} {'artif F1':>8}")

    orig = F.CRITERION_FEATURES
    try:
        for name, sets in VARIANTS.items():
            F.CRITERION_FEATURES = copy.deepcopy(sets)
            res = ev.evaluate(None, table=table, verbose=False, threshold_mode=args.mode)
            sc = res["scores"]
            qs = sc.get("spine: ИТОГ качество (ИЛИ критериев)")
            qf = sc.get("femur: ИТОГ качество (ИЛИ критериев)")
            fp = sc.get(f"femur: {LABEL_POSITIONING}")
            af = sc.get(f"spine: {LABEL_FOREIGN}")
            print(f"{name:40} {res['macro_f1']:8.3f} "
                  f"{(qs.auc if qs else float('nan')):7.3f} {(qf.auc if qf else float('nan')):7.3f} "
                  f"{(fp.f1 if fp else float('nan')):11.3f} {(af.f1 if af else float('nan')):8.3f}")
    finally:
        F.CRITERION_FEATURES = orig
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
