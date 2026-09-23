# -*- coding: utf-8 -*-
"""Сравнение стратегий выбора порога под ЧЕСТНОЙ схемой (dxa_real/evaluate).

Порог — самое слабое место при 6–10 положительных примерах: AUC оси 0.82, а F1
всего 0.24, т.е. ранжирование хорошее, а решение по порогу — плохое.

    python scripts/tune_threshold.py --root "...\\_dsroot"
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dxa_real import evaluate as ev            # noqa: E402
from dxa_real import features as F             # noqa: E402
from dxa_real import synth_real                # noqa: E402
from dxa_real.data import (LABEL_AXIS, LABEL_FOREIGN, LABEL_POSITIONING,  # noqa: E402
                           LABEL_ROI, load_dataset)

V3 = {
    ("spine", LABEL_AXIS): ["spine_midline_deg"],
    ("spine", LABEL_POSITIONING): ["spine_iliac_signal", "spine_bottom_cut"],
    ("spine", LABEL_FOREIGN): ["spine_ribs_signal", "spine_vertebra_peaks"],
    ("femur", LABEL_POSITIONING): ["femur_trochanter_bulge", "femur_troch_area_mm2",
                                   "femur_neck_width_mm"],
    ("femur", LABEL_ROI): ["femur_height_cm", "femur_bone_ratio", "femur_margin_min_cm"],
}


def grid_f1(lo, hi, n, metric="f1"):
    grid = np.linspace(lo, hi, n)

    def fn(y, p):
        if metric == "ba":
            scores = [balanced_accuracy_score(y, (p >= t).astype(int)) for t in grid]
        else:
            scores = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
        return float(grid[int(np.argmax(scores))])
    return fn


def report(name, res):
    qs = res["scores"].get("spine: ИТОГ качество (модель)")
    qf = res["scores"].get("femur: ИТОГ качество (модель)")
    print(f"{name:34} macroF1={res['macro_f1']:.3f}  spineQ={qs.auc:.3f}  femurQ={qf.auc:.3f}")
    return res["macro_f1"], {k: s.f1 for k, s in res["scores"].items() if "ИТОГ" not in k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    images = load_dataset(args.root)
    table = ev.build_table(images)
    F.CRITERION_FEATURES = copy.deepcopy(V3)

    variants = {
        "nested 0.05-0.95 (текущий)": (grid_f1(0.05, 0.95, 91), "nested"),
        "nested 0.01-0.99 (мелкая)": (grid_f1(0.01, 0.99, 197), "nested"),
        "prior (доля нарушений)": (ev._best_threshold, "prior"),
        "blend (nested + prior)": (ev._best_threshold, "blend"),
    }

    results = {}
    for name, (fn, mode) in variants.items():
        ev._best_threshold = fn
        t0 = time.time()
        res = ev.evaluate(images, table, verbose=False, threshold_mode=mode)
        m, per = report(name, res)
        results[name] = (m, per)
        print(f"    ({time.time()-t0:.0f}s)")

    print("\n" + "=" * 118)
    keys = sorted(next(iter(results.values()))[1].keys())
    short = [k.replace("spine: ", "S:").replace("femur: ", "F:") for k in keys]
    print(f"{'стратегия':34} {'macroF1':>8} " + " ".join(f"{s:>9}" for s in short))
    for name, (m, per) in results.items():
        print(f"{name:34} {m:8.3f} " + " ".join(f"{per[k]:9.3f}" for k in keys))


if __name__ == "__main__":
    main()
