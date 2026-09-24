# -*- coding: utf-8 -*-
"""Проверка источника для spine_positioning: cnn vs fused vs geo.

Использует реальные OOF-вероятности CNN (artifacts/oof_violation.npy) и
повторный StratifiedGroupKFold. Отвечает на вопрос: даёт ли смена источника
cnn->fused с новыми признаками честный прирост?
"""
from __future__ import annotations
import os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src.data import load_manifest, real_mask
from src.features import load_features

SEEDS = list(range(10)); N = 5

def clf():
    return Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(max_iter=1000, class_weight="balanced"))])

def run(mode, X, cnn, y, g):
    aucs, f1s = [], []
    for seed in SEEDS:
        oof = np.full(len(y), np.nan)
        for tr, va in StratifiedGroupKFold(n_splits=N, shuffle=True, random_state=seed).split(X, y, groups=g):
            if len(np.unique(y[tr])) < 2:
                continue
            if mode == "cnn":
                oof[va] = cnn[va]
            else:
                Xf = np.column_stack([cnn, X]) if mode == "fused" else X
                m = clf().fit(Xf[tr], y[tr]); oof[va] = m.predict_proba(Xf[va])[:, 1]
        ok = np.isfinite(oof)
        if len(np.unique(y[ok])) < 2:
            continue
        aucs.append(roc_auc_score(y[ok], oof[ok]))
        f1s.append(f1_score(y[ok], (oof[ok] >= 0.5).astype(int), zero_division=0))
    return np.mean(aucs), np.std(aucs), np.mean(f1s), np.std(f1s)

def main():
    man = load_manifest(C.MANIFEST_CSV)
    df = load_features(man)
    oof_v = np.load(C.OOF_VIOLATION_NPY)
    real = real_mask(df)
    j = C.VIOLATION_IDX["spine_positioning"]
    sub = df[df["region"] == C.REGION_SPINE].reset_index(drop=True)
    idx = df.index[df["region"] == C.REGION_SPINE].values
    y = sub["viol_spine_positioning"].astype(float).values
    g = sub["study_uid"].values
    cnn = oof_v[idx, j]
    cols_new = ["spine_iliac_signal", "art_max_out", "atlas_outlier_area", "bone_eccentricity"]
    cols_old = ["spine_iliac_signal", "spine_bottom_cut"]
    out = []
    for mode, cols, tag in [
        ("cnn", None, "cnn (v3)"),
        ("geo", cols_old, "geo base (v3 features)"),
        ("fused", cols_old, "fused base"),
        ("geo", cols_new, "geo new (4)"),
        ("fused", cols_new, "fused new (4)"),
    ]:
        X = sub[cols].astype(float).fillna(0).values if cols else np.zeros((len(y), 0))
        a, a_s, f, f_s = run(mode, X, cnn, y, g)
        out.append("  %-26s AUC=%.3f±%.3f  macroF1=%.3f±%.3f" % (tag, a, a_s, f, f_s))
    txt = "spine_positioning: cnn vs fused vs geo\n" + "\n".join(out)
    open("positioning_source.txt", "w", encoding="utf-8").write(txt)
    print(txt)

if __name__ == "__main__":
    main()