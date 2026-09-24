# -*- coding: utf-8 -*-
"""Честное сравнение наборов признаков для «предметов» и «укладки».

В отличие от жадного отбора (который выбирает признаки на том же CV, где потом
измеряет), здесь сравниваются ЗАРАНЕЕ заданные наборы на повторном
StratifiedGroupKFold: 10 повторов x 5 фолдов, группы = study_uid. Это даёт
устойчивую оценку (mean ± std) и защищает от подгонки под разбиение.

    python scripts/ablate_criteria.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C  # noqa: E402
from src.data import load_manifest  # noqa: E402
from src.features import load_features  # noqa: E402

SEEDS = list(range(10))
N_SPLITS = 5

CANDS = {
    "spine_artifacts": {
        "base (v3)": ["spine_ribs_signal", "spine_vertebra_peaks"],
        "art_edge_out_p99 (1)": ["art_edge_out_p99"],
        "art (2)": ["art_edge_out_p99", "atlas_resid_out_p99"],
        "art+atlas (4)": ["art_edge_out_p99", "atlas_bone_iou",
                          "atlas_resid_out_p99", "art_bright_out_edge"],
        "phys+art (4)": ["spine_ribs_signal", "spine_vertebra_peaks",
                         "art_edge_out_p99", "atlas_resid_out_p99"],
    },
    "spine_positioning": {
        "base (v3)": ["spine_iliac_signal", "spine_bottom_cut"],
        "iliac+atlas+exc (3)": ["spine_iliac_signal", "atlas_outlier_area",
                                "bone_eccentricity"],
        "iliac+art+atlas+exc (4)": ["spine_iliac_signal", "art_max_out",
                                    "atlas_outlier_area", "bone_eccentricity"],
        "atlas geo (3)": ["atlas_bone_iou", "align_angle", "spine_iliac_signal"],
        "atlas_resid (2)": ["atlas_outlier_area", "atlas_bone_iou"],
    },
}


def _clf():
    return Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(max_iter=1000,
                                              class_weight="balanced", C=1.0))])


def evaluate(X, y, groups, seeds=SEEDS):
    aucs, f1s = [], []
    for seed in seeds:
        oof = np.full(len(y), np.nan)
        sgk = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for tr, va in sgk.split(X, y, groups=groups):
            if len(np.unique(y[tr])) < 2:
                continue
            m = _clf().fit(X[tr], y[tr])
            oof[va] = m.predict_proba(X[va])[:, 1]
        ok = np.isfinite(oof)
        if len(np.unique(y[ok])) < 2:
            continue
        aucs.append(roc_auc_score(y[ok], oof[ok]))
        f1s.append(f1_score(y[ok], (oof[ok] >= 0.5).astype(int), zero_division=0))
    return (float(np.mean(aucs)), float(np.std(aucs)),
            float(np.mean(f1s)), float(np.std(f1s)))


def main():
    man = load_manifest(C.MANIFEST_CSV)
    df = load_features(man)
    out = []
    for tgt, sets in CANDS.items():
        sub = df[df["region"] == C.REGION_SPINE].reset_index(drop=True)
        y = sub["viol_" + tgt].astype(float).values
        g = sub["study_uid"].values
        out.append("\n=== %s (положительных %d/%d) ===" % (tgt, int(y.sum()), len(y)))
        rows = []
        for name, cols in sets.items():
            cols = [c for c in cols if c in sub.columns]
            X = sub[cols].astype(float).fillna(0.0).values
            a_m, a_s, f_m, f_s = evaluate(X, y, g)
            rows.append((name, len(cols), a_m, a_s, f_m, f_s, cols))
            out.append("  %-26s k=%d  AUC=%.3f±%.3f  macroF1=%.3f±%.3f"
                       % (name, len(cols), a_m, a_s, f_m, f_s))
        best = max(rows, key=lambda r: r[2])
        out.append("  --> по AUC лучший: %s" % best[0])
    txt = "\n".join(out)
    with open("ablate_out.txt", "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()