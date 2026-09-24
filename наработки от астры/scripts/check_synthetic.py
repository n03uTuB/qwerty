# -*- coding: utf-8 -*-
"""Проверка качества синтетики: похожа ли она на настоящие нарушения.

Две опасности синтетики:
  1) она тривиально отделима (CNN выучит след преобразования, а не нарушение);
  2) её величина не совпадает с реальными нарушениями (задача станет нереально лёгкой
     или наоборот).
Здесь сравниваем ключевые признаки по группам: реальная норма, реальное нарушение,
синтетическое нарушение, контроль (та же обработка, метка 0).

    cd dxa_qc && python ../scripts/check_synthetic.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dxa_qc"))

from src import config as C   # noqa: E402

# критерий -> (область, колонка манифеста, ключевой признак)
CHECKS = [
    ("spine_axis", "spine", "spine_midline_angle"),
    ("spine_positioning", "spine", "spine_iliac_signal"),
    ("femur_roi", "femur", "femur_height_cm"),
]


def main() -> None:
    manifest = pd.read_csv(C.MANIFEST_CSV)
    feat = pd.read_csv(os.path.join(C.ARTIFACTS_DIR, "features.csv"))
    keep = ["image_uid"] + [c for c in feat.columns
                            if c not in ("study_uid", "region", "image_uid")]
    df = manifest.merge(feat[keep], on="image_uid", how="left")

    print(f"строк: {len(df)}  синтетических: {int(df.synthetic.sum())}\n")
    for name, region, key in CHECKS:
        sub = df[df.region.str.contains(region)]
        col = "viol_" + name
        groups = {
            "реальная норма": sub[(~sub.synthetic) & (sub[col] == 0)][key],
            "реальное нарушение": sub[(~sub.synthetic) & (sub[col] == 1)][key],
            "синтетика (нарушение)": sub[sub.synthetic & (sub[col] == 1)][key],
            "контроль (обработка, метка 0)": sub[sub.synthetic & (sub[col] == 0)][key],
        }
        print(f"=== {name} :: {key} ===")
        for label, vals in groups.items():
            vals = vals.dropna().values
            if len(vals) == 0:
                print(f"    {label:32} n=0")
                continue
            print(f"    {label:32} n={len(vals):3d}  медиана={np.median(vals):8.3f}  "
                  f"среднее={vals.mean():8.3f}  ст.откл={vals.std():7.3f}")
        print()

    # контроль отделимости: AUC «синтетическое нарушение vs контроль» по признакам критерия
    from sklearn.metrics import roc_auc_score
    print("=== отделимость синтетики от контроля (AUC по признакам критерия) ===")
    for name, cols in C_CRITERION.items():
        region = "spine" if name.startswith("spine") else "femur"
        sub = df[df.region.str.contains(region)]
        pos = sub[sub.synthetic & (sub["viol_" + name] == 1)]
        neg = sub[sub.synthetic & (sub["viol_" + name] == 0)]
        if len(pos) < 5 or len(neg) < 5:
            continue
        detail = []
        for c in cols:
            if c not in df.columns:
                continue
            x = np.concatenate([pos[c].values, neg[c].values]).astype(float)
            y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
            ok = ~np.isnan(x)
            if len(np.unique(y[ok])) < 2:
                continue
            a = roc_auc_score(y[ok], x[ok])
            detail.append(f"{c}={max(a, 1 - a):.3f}")
        print(f"    {name:22} n_pos={len(pos):3d} n_neg={len(neg):3d}  " + "  ".join(detail))


C_CRITERION = {
    "spine_axis": ["spine_midline_angle"],
    "spine_positioning": ["spine_iliac_signal", "spine_bottom_cut"],
    "spine_artifacts": ["spine_ribs_signal", "spine_vertebra_peaks"],
    "femur_positioning": ["femur_trochanter_bulge", "femur_troch_area_mm2", "femur_neck_width_mm"],
    "femur_roi": ["femur_height_cm", "femur_bone_ratio", "femur_margin_min_cm"],
}

if __name__ == "__main__":
    main()
