# -*- coding: utf-8 -*-
"""Сравнение наборов признаков под ЧЕСТНОЙ кросс-валидацией (dxa_real/evaluate).

Выбор признаков проверяется на той же вложенной CV, что и отчётные метрики:
порог подбирается внутри обучающей части, доверительные интервалы — по исследованиям.

    python scripts/compare_feature_sets.py --root "...\\_dsroot"
    python scripts/compare_feature_sets.py --root ... --synthetic
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

# --- текущий набор (как в проекте) ---
V0 = copy.deepcopy(F.CRITERION_FEATURES)

# --- предлагаемый набор: минимальные семантически прямые измерения критерия ТЗ ---
V1 = {
    ("spine", LABEL_AXIS): ["spine_midline_deg"],
    ("spine", LABEL_POSITIONING): ["spine_iliac_signal", "spine_bottom_cut"],
    ("spine", LABEL_FOREIGN): ["spine_ribs_signal", "spine_vertebra_peaks"],
    ("femur", LABEL_POSITIONING): ["femur_neck_width_mm", "femur_margin_lateral_cm",
                                   "femur_shaft_width_mm"],
    ("femur", LABEL_ROI): ["femur_height_cm", "femur_bone_ratio", "femur_margin_min_cm"],
}

# --- вариант с метаданными выгрузки (число копий снимка) ---
V2 = copy.deepcopy(V1)
V2[("femur", LABEL_POSITIONING)] = V1[("femur", LABEL_POSITIONING)] + ["copies"]
V2[("spine", LABEL_POSITIONING)] = V1[("spine", LABEL_POSITIONING)] + ["copies"]


def run(name, features_map, images, table, use_meta=False):
    F.CRITERION_FEATURES = features_map
    t0 = time.time()
    res = ev.evaluate(images, table, verbose=False, use_meta=use_meta)
    macro = res["macro_f1"]
    print(f"\n### {name}  (macro-F1={macro:.3f}, {time.time()-t0:.0f}s)")
    for key in sorted(res["scores"]):
        s = res["scores"][key]
        print(f"    {key:42} n={s.n:3d} pos={s.positives:2d}  "
              f"AUC={s.auc:.3f}  F1={s.f1:.3f}  чувст={s.sensitivity:.2f}  спец={s.specificity:.2f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--per-type", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    images = load_dataset(args.root)
    if args.synthetic:
        rng = np.random.default_rng(args.seed)
        images = images + synth_real.build(images, rng, per_type=args.per_type)
        print(synth_real.summary(images[-1:]) if False else f"[data] +синтетика, всего {len(images)}")

    table = ev.build_table(images)
    print(f"[data] изображений: {len(images)}  (синтетика: {int(table['synthetic'].sum())})")

    results = {}
    for name, fmap in (("V0 (текущий)", V0), ("V1 (минимальный)", V1), ("V2 (+copies)", V2)):
        results[name] = run(name, copy.deepcopy(fmap), images, table)

    print("\n" + "=" * 70)
    print(f"{'набор':20} {'macro-F1':>9}  {'quality spine':>13} {'quality femur':>13}")
    for name, res in results.items():
        qs = res["scores"].get("spine: ИТОГ качество (ИЛИ критериев)")
        qf = res["scores"].get("femur: ИТОГ качество (ИЛИ критериев)")
        qs_auc = f"{qs.auc:.3f}" if qs else "n/a"
        qf_auc = f"{qf.auc:.3f}" if qf else "n/a"
        print(f"{name:20} {res['macro_f1']:9.3f}  {qs_auc:>13} {qf_auc:>13}")


if __name__ == "__main__":
    main()
