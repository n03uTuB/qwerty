# -*- coding: utf-8 -*-
"""Что даёт CNN поверх геометрии: по каждому критерию три оценки.

    CNN-only  — OOF-вероятности сети (artifacts/oof_violation.npy);
    геометрия — логистическая регрессия на признаках критерия (групповая CV);
    гибрид    — OOF стекера (artifacts/stack_oof_violation.npy).

Считается только по реальным снимкам. Запуск после train + stack:
    cd dxa_qc && python ../scripts/compare_cnn_vs_geometry.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dxa_qc"))

from src import config as C          # noqa: E402
from src import dataset as ds        # noqa: E402
from src import stack as st          # noqa: E402
from src import train as tr          # noqa: E402


def _cv_geo(X, y, groups, seeds=(0, 1, 2, 3, 4)):
    probs = np.zeros(len(y))
    cnt = np.zeros(len(y))
    for seed in seeds:
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for trn, va in skf.split(X, y, groups):
            m = Pipeline([("s", StandardScaler()),
                          ("c", LogisticRegression(max_iter=2000,
                                                   class_weight="balanced", C=0.5))])
            m.fit(X[trn], y[trn])
            probs[va] += m.predict_proba(X[va])[:, 1]
            cnt[va] += 1
    ok = cnt > 0
    probs[ok] /= cnt[ok]
    return probs, ok


def _f1(y, p, mode=None):
    thr, _ = tr.pick_threshold(y, p, mode)
    return float(f1_score(y, (p >= thr).astype(int), zero_division=0))


def main() -> None:
    manifest = ds.load_manifest(C.MANIFEST_CSV)
    feat = pd.read_csv(os.path.join(C.ARTIFACTS_DIR, "features.csv"))
    keep = ["image_uid"] + [c for c in feat.columns
                            if c not in ("study_uid", "region", "image_uid")]
    data = manifest.merge(feat[keep], on="image_uid", how="left")
    real = ds.real_mask(data).values

    oof_cnn = np.load(os.path.join(C.ARTIFACTS_DIR, "oof_violation.npy"))
    fused_path = os.path.join(C.ARTIFACTS_DIR, "stack_oof_violation.npy")
    oof_fused = np.load(fused_path) if os.path.isfile(fused_path) else np.full_like(oof_cnn, np.nan)

    print(f"{'критерий':20} {'n':>4} {'pos':>4} | {'CNN AUC/F1':>16} {'геом AUC/F1':>16} "
          f"{'гибрид AUC/F1':>16}")
    rows = {"cnn": [], "geo": [], "fused": []}
    for j, name in enumerate(C.VIOLATIONS):
        region = "spine" if name.startswith("spine") else "femur"
        sub = data[data.region.str.contains(region)]
        y = sub["viol_" + name].values.astype(float)
        known = ~np.isnan(y)
        yy = y[known].astype(int)
        g = sub.loc[known, "study_uid"].values
        idx = np.where(known)[0]

        cols = st.CRITERION_FEATURES.get(name, [])
        X = sub.loc[known, cols].fillna(0).values if cols else np.zeros((len(yy), 1))

        p_cnn = oof_cnn[idx, j]
        p_geo, ok_geo = _cv_geo(X, yy, g)
        p_fused = oof_fused[idx, j]

        def line(p):
            ok = ~np.isnan(p)
            if ok.sum() < 5 or len(np.unique(yy[ok])) < 2:
                return "n/a", None
            auc = roc_auc_score(yy[ok], p[ok])
            f1 = _f1(yy[ok], p[ok])
            return f"{auc:.3f}/{f1:.3f}", (auc, f1)

        s_cnn, m_cnn = line(p_cnn)
        s_geo, m_geo = line(p_geo)
        s_fused, m_fused = line(p_fused)
        for key, mm in (("cnn", m_cnn), ("geo", m_geo), ("fused", m_fused)):
            if mm:
                rows[key].append(mm[1])
        print(f"{name:20} {len(yy):4d} {int(yy.sum()):4d} | {s_cnn:>16} {s_geo:>16} {s_fused:>16}")

    print("\nmacro-F1 по критериям:")
    for key, label in (("cnn", "только CNN"), ("geo", "только геометрия"), ("fused", "гибрид")):
        vals = rows[key]
        print(f"    {label:20} {np.mean(vals) if vals else float('nan'):.3f}")


if __name__ == "__main__":
    main()
