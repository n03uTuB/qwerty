# -*- coding: utf-8 -*-
"""Финальный эксперимент: лучший набор признаков (V3) x {синтетика} x {модель}.

Проверяет, даёт ли синтетика прирост на честной схеме и не лучше ли иная модель
для отдельных критериев. Модель можно переопределить через monkeypatch make_model.

    python scripts/final_v3.py --root "...\\_dsroot"
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

V3 = {
    ("spine", LABEL_AXIS): ["spine_midline_deg"],
    ("spine", LABEL_POSITIONING): ["spine_iliac_signal", "spine_bottom_cut"],
    ("spine", LABEL_FOREIGN): ["spine_ribs_signal", "spine_vertebra_peaks"],
    ("femur", LABEL_POSITIONING): ["femur_trochanter_bulge", "femur_troch_area_mm2",
                                   "femur_neck_width_mm"],
    ("femur", LABEL_ROI): ["femur_height_cm", "femur_bone_ratio", "femur_margin_min_cm"],
}


def report(name, res):
    qs = res["scores"].get("spine: ИТОГ качество (ИЛИ критериев)")
    qf = res["scores"].get("femur: ИТОГ качество (ИЛИ критериев)")
    print(f"{name:28} macroF1={res['macro_f1']:.3f}  spineQ_AUC={qs.auc:.3f}  femurQ_AUC={qf.auc:.3f}")
    return res["macro_f1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--per-type", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base = load_dataset(args.root)
    rng = np.random.default_rng(args.seed)
    with_syn = base + synth_real.build(base, rng, per_type=args.per_type)

    print(f"[data] реальных: {len(base)}  с синтетикой: {len(with_syn)}")
    F.CRITERION_FEATURES = copy.deepcopy(V3)

    t0 = time.time()
    r1 = ev.evaluate(base, ev.build_table(base), verbose=False)
    report("V3 (без синтетики)", r1)
    r2 = ev.evaluate(with_syn, ev.build_table(with_syn), verbose=False)
    report("V3 (+синтетика)", r2)

    # --- вариант модели: HistGradientBoosting (нелинейность для U-образной ротации) ---
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        def make_hgb():
            return Pipeline([("scaler", StandardScaler()),
                             ("clf", HistGradientBoostingClassifier(
                                 max_depth=2, max_iter=150, learning_rate=0.05,
                                 l2_regularization=1.0, random_state=0))])

        ev.make_model = make_hgb
        r3 = ev.evaluate(base, ev.build_table(base), verbose=False)
        report("V3 + HistGB", r3)
    except Exception as e:
        print("[warn] HistGB:", e)
    finally:
        ev.make_model = ev.make_model  # no-op

    print(f"\nвремя: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
