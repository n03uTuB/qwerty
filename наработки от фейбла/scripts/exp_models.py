# -*- coding: utf-8 -*-
"""Модельный зоопарк по геометрическим признакам: подбор лучшей модели+набора
под каждый критерий на ЧЕСТНОЙ схеме (StratifiedGroupKFold по study_uid,
вложенный порог, 5 сидов).

Сохраняет лучшие OOF-вероятности по критерию в out/geo_oof_<crit>.npy
(для последующего фьюжна с CNN).

Запуск:
    python scripts/exp_models.py [--mode nested|prior]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlab as M  # noqa: E402
from dxa_real.data import (LABEL_AXIS, LABEL_FOREIGN, LABEL_POSITIONING,  # noqa: E402
                           LABEL_ROI)

CACHE = os.path.join("out", "feat_cache.csv")
OUTDIR = "out"

# ---- наборы признаков-кандидатов под каждый критерий --------------------- #
SETS = {
    ("femur", LABEL_POSITIONING): {
        "V3": ["femur_trochanter_bulge", "femur_troch_area_mm2", "femur_neck_width_mm"],
        "univ3": ["femur_shaft_deg", "femur_troch_position", "femur_margin_lateral_cm"],
        "univ3_sq": ["femur_shaft_deg_sq", "femur_troch_position", "femur_margin_lateral_cm"],
        "dlr2": ["dlr_femur_bone_ratio", "femur_shaft_deg"],
        "margins": ["femur_margin_min_cm", "femur_width_cm"],
        "broad": ["femur_shaft_deg", "femur_troch_position", "femur_margin_lateral_cm",
                  "femur_bone_ratio", "femur_troch_peak_mm", "femur_head_diameter_mm",
                  "femur_ischium_signal"],
        "all": None,  # все femur_/dlr_ признаки
    },
    ("femur", LABEL_ROI): {
        "V3": ["femur_height_cm", "femur_bone_ratio", "femur_margin_min_cm"],
        "h2": ["femur_height_cm", "femur_bone_ratio"],
        "h_marg": ["femur_height_cm", "femur_margin_min_cm", "femur_margin_lateral_cm"],
        "frame": ["rows", "cols", "femur_height_cm", "femur_width_cm"],
        "all": None,
    },
    ("spine", LABEL_FOREIGN): {
        "V3": ["spine_ribs_signal", "spine_vertebra_peaks"],
        "V3res": ["spine_ribs_signal", "spine_vertebra_peaks", "spine_midline_residual"],
        "ribs": ["spine_ribs_signal"],
        "ribs_res": ["spine_ribs_signal", "spine_midline_residual"],
        "all": None,
    },
    ("spine", LABEL_AXIS): {
        "V3": ["spine_midline_deg"],
        "mid_axis": ["spine_midline_deg", "spine_axis_deg"],
        "mid_res": ["spine_midline_deg", "spine_midline_residual"],
        "all": None,
    },
    ("spine", LABEL_POSITIONING): {
        "V3": ["spine_iliac_signal", "spine_bottom_cut"],
        "v3m": ["spine_iliac_signal", "spine_bottom_cut", "spine_margin_bottom_cm"],
        "v3t": ["spine_iliac_signal", "spine_bottom_cut", "spine_top_cut",
                "spine_margin_top_cm"],
        "all": None,
    },
}


def zoo(fast: bool = False):
    from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                                  HistGradientBoostingClassifier, RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    Z = {}
    for c in (0.1, 0.5, 1.0, 2.0):
        Z[f"LR C={c}"] = (lambda c=c: Pipeline([("s", StandardScaler()),
                         ("c", LogisticRegression(max_iter=2000, class_weight="balanced", C=c))]))
    Z["RF"] = (lambda: RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                              random_state=0, n_jobs=-1))
    Z["ET"] = (lambda: ExtraTreesClassifier(n_estimators=400, class_weight="balanced_subsample",
                                            random_state=0, n_jobs=-1))
    Z["HGB"] = lambda: HistGradientBoostingClassifier(random_state=0)
    if fast:
        return Z
    Z["SVC rbf"] = (lambda: Pipeline([("s", StandardScaler()),
                    ("c", SVC(probability=True, class_weight="balanced", C=1.0, random_state=0))]))
    Z["SVC lin"] = (lambda: Pipeline([("s", StandardScaler()),
                    ("c", SVC(kernel="linear", probability=True, class_weight="balanced",
                              C=1.0, random_state=0))]))
    Z["GB"] = lambda: GradientBoostingClassifier(random_state=0)
    Z["KNN5"] = (lambda: Pipeline([("s", StandardScaler()), ("c", KNeighborsClassifier(5))]))
    Z["MLP16"] = (lambda: Pipeline([("s", StandardScaler()),
                   ("c", MLPClassifier(hidden_layer_sizes=(16,), max_iter=3000, random_state=0))]))
    return Z


def columns_for(table, region, cols):
    if cols is not None:
        return list(cols)
    if region == "femur":
        return [c for c in table.columns if c.startswith(("femur_", "dlr_"))]
    return [c for c in table.columns if c.startswith("spine_")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="nested", choices=["nested", "prior"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    table = pd.read_csv(CACHE)
    print(f"[models] таблица: {len(table)} строк, режим={args.mode}, сидов={args.seeds}")
    Z = zoo(fast=args.fast)
    seeds = tuple(range(args.seeds))
    summary = {}
    for (region, label), sets in SETS.items():
        m = (table["region"] == region).values
        y = table.loc[m, label].values.astype(float)
        known = ~np.isnan(y)
        if known.sum() == 0 or len(np.unique(y[known])) < 2:
            print(f"\n### {region}/{label}: пропуск (нет данных)")
            continue
        idx = np.where(m)[0][known]
        yk = y[known].astype(int)
        gk = table.loc[m, "study_uid"].values[known]
        print(f"\n### {region}/{label}  (n={len(yk)}, поз={int(yk.sum())})")
        rows = []
        for set_name, cols in sets.items():
            use = columns_for(table, region, cols)
            X = table.loc[m, use].fillna(0.0).values.astype(float)[known]
            for model_name, factory in Z.items():
                probs, preds, _ = M.cv_oof(X, yk, gk, factory, seeds=seeds,
                                           threshold_mode=args.mode)
                mt = M.metrics(yk, probs, preds, gk)
                rows.append((mt["auc"], mt["ap"], mt["f1"], set_name, model_name,
                             probs, preds))
        rows.sort(key=lambda r: (-np.nan_to_num(r[0], nan=-1), -np.nan_to_num(r[1], nan=-1)))
        for auc, ap_, f1, set_name, model_name, probs, preds in rows[:10]:
            print(f"   {set_name:10} {model_name:10} AUC={auc:.3f} AP={ap_:.3f} F1={f1:.3f}")
        # сохранить лучший по AUC
        best = rows[0]
        np.save(os.path.join(OUTDIR, f"geo_oof_{region}_{label}.npy"), best[5])
        np.save(os.path.join(OUTDIR, f"geo_pred_{region}_{label}.npy"), best[6])
        summary[(region, label)] = (best[3], best[4], best[0], best[2])
    print("\n=== ЛУЧШЕЕ по критерию (набор, модель, AUC, F1) ===")
    for k, v in summary.items():
        print(f"  {k[0]:6} {k[1]:34} set={v[0]:10} model={v[1]:10} AUC={v[2]:.3f} F1={v[3]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
