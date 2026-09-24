# -*- coding: utf-8 -*-
"""Сравнение источников и наборов признаков для femur_positioning (реальные данные)."""
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
            if len(np.unique(y[tr])) < 2: continue
            if mode == "cnn":
                oof[va] = cnn[va]
            else:
                Xf = np.column_stack([cnn, X]) if mode == "fused" else X
                m = clf().fit(Xf[tr], y[tr]); oof[va] = m.predict_proba(Xf[va])[:,1]
        ok = np.isfinite(oof)
        if len(np.unique(y[ok])) < 2: continue
        aucs.append(roc_auc_score(y[ok], oof[ok]))
        f1s.append(f1_score(y[ok], (oof[ok]>=0.5).astype(int), zero_division=0))
    return np.mean(aucs), np.std(aucs), np.mean(f1s), np.std(f1s)

def main():
    man = load_manifest(C.MANIFEST_CSV)
    df = load_features(man)
    oof_v = np.load(C.OOF_VIOLATION_NPY)
    j = C.VIOLATION_IDX["femur_positioning"]
    m = df["region"].isin([C.REGION_FEMUR_LEFT, C.REGION_FEMUR_RIGHT]).values & df["viol_femur_positioning"].notna().values
    sub = df[m].reset_index(drop=True)
    idx = df.index[m].values
    y = sub["viol_femur_positioning"].astype(float).values
    g = sub["study_uid"].values
    cnn = oof_v[idx, j]
    sets = {
        "cur O: min+width": ["femur_margin_min_cm", "femur_width_cm"],
        "troch_bulge (1)": ["femur_trochanter_bulge"],
        "width+bulge": ["femur_width_cm", "femur_trochanter_bulge"],
        "width+bulge+neck": ["femur_width_cm", "femur_trochanter_bulge", "femur_neck_width_mm"],
        "width+bulge+shaft": ["femur_width_cm", "femur_trochanter_bulge", "femur_shaft_deg"],
        "margin+bulge+neck": ["femur_margin_min_cm", "femur_trochanter_bulge", "femur_neck_width_mm"],
        "A troch+area+neck": ["femur_trochanter_bulge", "femur_troch_area_mm2", "femur_neck_width_mm"],
        "width+bulge+axis": ["femur_width_cm", "femur_trochanter_bulge", "femur_axis_angle"],
    }
    out = ["=== femur_positioning: источники/наборы (AUC±std, macroF1±std) ==="]
    a,s,f,fs = run("cnn", np.zeros((len(y),0)), cnn, y, g)
    out.append("  %-26s AUC=%.3f±%.3f F1=%.3f±%.3f" % ("cnn (v3)", a,s,f,fs))
    for name, cols in sets.items():
        X = sub[cols].astype(float).fillna(0).values
        for mode in ("geo","fused"):
            a,s,f,fs = run(mode, X, cnn, y, g)
            out.append("  %-14s %-26s AUC=%.3f±%.3f F1=%.3f±%.3f" % (mode, name, a,s,f,fs))
    txt = "\n".join(out)
    open("femur_sets.txt","w",encoding="utf-8").write(txt)
    print(txt.encode("ascii","replace").decode())

if __name__ == "__main__":
    main()