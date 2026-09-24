# -*- coding: utf-8 -*-
"""Разведка признаков femur_positioning на реальных данных."""
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
from src.features import load_features, FEATURE_NAMES

SEEDS = list(range(10)); N = 5

def clf():
    return Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(max_iter=1000, class_weight="balanced"))])

def cv1(X, y, g):
    aucs = []
    for seed in SEEDS:
        oof = np.full(len(y), np.nan)
        for tr, va in StratifiedGroupKFold(n_splits=N, shuffle=True, random_state=seed).split(X, y, groups=g):
            if len(np.unique(y[tr])) < 2: continue
            m = clf().fit(X[tr], y[tr]); oof[va] = m.predict_proba(X[va])[:,1]
        ok = np.isfinite(oof)
        if len(np.unique(y[ok])) < 2: continue
        aucs.append(roc_auc_score(y[ok], oof[ok]))
    return float(np.mean(aucs)), float(np.std(aucs))

def main():
    man = load_manifest(C.MANIFEST_CSV)
    df = load_features(man)
    real = real_mask(df)
    # обе стороны бедра вместе
    sub = df[df["region"].isin([C.REGION_FEMUR_LEFT, C.REGION_FEMUR_RIGHT])].copy()
    sub = sub[sub["viol_femur_positioning"].notna()].reset_index(drop=True)
    y = sub["viol_femur_positioning"].astype(float).values
    g = sub["study_uid"].values
    print("femur_positioning pos=%d/%d" % (int(y.sum()), len(y)))
    cands = [c for c in FEATURE_NAMES if c in sub.columns]
    singles = []
    for c in cands:
        X = sub[[c]].astype(float).fillna(0).values
        a, s = cv1(X, y, g)
        singles.append((abs(a-0.5)+0.5, a, s, c))
    singles.sort(reverse=True)
    out = ["=== одиночные признаки femur_positioning (AUC, 10x5) ==="]
    for _, a, s, c in singles[:25]:
        out.append("  %-26s AUC=%.3f±%.3f" % (c, a, s))
    # текущий набор
    cur = ["femur_margin_min_cm", "femur_width_cm"]
    a, s = cv1(sub[cur].astype(float).fillna(0).values, y, g)
    out.append("\nтекущий набор %s: AUC=%.3f±%.3f" % (cur, a, s))
    # по сторонам отдельно
    for reg in [C.REGION_FEMUR_LEFT, C.REGION_FEMUR_RIGHT]:
        r = sub[sub["region"]==reg].reset_index(drop=True)
        yy = r["viol_femur_positioning"].astype(float).values
        out.append("%s: pos=%d/%d" % (reg, int(yy.sum()), len(yy)))
    txt = "\n".join(out)
    open("femur_explore.txt","w",encoding="utf-8").write(txt)
    print(txt.encode("ascii","replace").decode())

if __name__ == "__main__":
    main()