# -*- coding: utf-8 -*-
"""Перебор признаков для самого трудного критерия — укладка/ротация бедра.

Остальные критерии зафиксированы на лучшем наборе (V1). Для каждого варианта
печатаются честные AUC/F1 самого критерия и итогового качества бедра (ИЛИ критериев).

    python scripts/tune_femur_positioning.py --root "...\\_dsroot"
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
from dxa_real import synth_real                # noqa: E402
from dxa_real.data import (LABEL_AXIS, LABEL_FOREIGN, LABEL_POSITIONING,  # noqa: E402
                           LABEL_ROI, load_dataset)

BASE = {
    ("spine", LABEL_AXIS): ["spine_midline_deg"],
    ("spine", LABEL_POSITIONING): ["spine_iliac_signal", "spine_bottom_cut"],
    ("spine", LABEL_FOREIGN): ["spine_ribs_signal", "spine_vertebra_peaks"],
    ("femur", LABEL_ROI): ["femur_height_cm", "femur_bone_ratio", "femur_margin_min_cm"],
}

CANDIDATES = {
    "A troch+area+neck (текущий)": ["femur_trochanter_bulge", "femur_troch_area_mm2",
                                     "femur_neck_width_mm"],
    "B troch+neck": ["femur_trochanter_bulge", "femur_neck_width_mm"],
    "L margins+bulge": ["femur_margin_min_cm", "femur_margin_top_cm",
                        "femur_trochanter_bulge"],
    "O margins+width": ["femur_margin_min_cm", "femur_width_cm"],
    "S A+L (комбинированный)": ["femur_trochanter_bulge", "femur_troch_area_mm2",
                                "femur_neck_width_mm", "femur_margin_min_cm",
                                "femur_margin_top_cm"],
    "T troch+neck+margin": ["femur_trochanter_bulge", "femur_neck_width_mm",
                            "femur_margin_min_cm"],
    "U width+neck": ["femur_width_cm", "femur_neck_width_mm"],
    "V width+bulge": ["femur_width_cm", "femur_trochanter_bulge"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--per-type", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", default="nested",
                    choices=["nested", "prior", "blend", "f1"],
                    help="режим выбора порога (важно: в пайплайне по умолчанию prior)")
    args = ap.parse_args()

    images = load_dataset(args.root)
    if args.synthetic:
        rng = np.random.default_rng(args.seed)
        images = images + synth_real.build(images, rng, per_type=args.per_type)
    table = ev.build_table(images)

    rows = []
    for name, cols in CANDIDATES.items():
        fmap = copy.deepcopy(BASE)
        fmap[("femur", LABEL_POSITIONING)] = cols
        F.CRITERION_FEATURES = fmap
        t0 = time.time()
        res = ev.evaluate(images, table, verbose=False, threshold_mode=args.mode)
        pos = res["scores"].get("femur: Некорректная укладка")
        qf = res["scores"].get("femur: ИТОГ качество (ИЛИ критериев)")
        rows.append((name, res["macro_f1"], pos.auc, pos.f1, qf.auc, qf.f1))
        print(f"{name:26} macroF1={res['macro_f1']:.3f}  pos AUC={pos.auc:.3f} F1={pos.f1:.3f}"
              f"  | femurQ AUC={qf.auc:.3f} F1={qf.f1:.3f}  ({time.time()-t0:.0f}s)")

    print("\n" + "=" * 100)
    print(f"{'вариант':26} {'macroF1':>8} {'pos AUC':>8} {'pos F1':>7} {'femurQ AUC':>11} {'femurQ F1':>10}")
    for name, m, a, f, qa, qf in sorted(rows, key=lambda r: -r[1]):
        print(f"{name:26} {m:8.3f} {a:8.3f} {f:7.3f} {qa:11.3f} {qf:10.3f}")


if __name__ == "__main__":
    main()
