# -*- coding: utf-8 -*-
"""Диагностика по всем критериям на уже посчитанных признаках (быстро, без DICOM).

Для каждого критерия печатает:
  * текущий набор признаков и его честную групповую CV-AUC;
  * топ одиночных признаков (кандидаты на замену/добавление).

Только реальные снимки. Запуск:
    cd dxa_qc && python ../scripts/diag_all_criteria.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dxa_qc"))

from src import config as C      # noqa: E402
from src import dataset as ds    # noqa: E402
from src import stack as st      # noqa: E402


def cv_auc(X, y, groups, seeds=(0, 1, 2, 3, 4)):
    probs = np.zeros(len(y))
    cnt = np.zeros(len(y))
    for seed in seeds:
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for trn, va in skf.split(X, y, groups):
            m = Pipeline([("s", StandardScaler()),
                          ("c", LogisticRegression(max_iter=3000, class_weight="balanced",
                                                   C=0.5))])
            m.fit(X[trn], y[trn])
            probs[va] += m.predict_proba(X[va])[:, 1]
            cnt[va] += 1
    ok = cnt > 0
    return roc_auc_score(y[ok], probs[ok] / cnt[ok])


def main() -> None:
    manifest = ds.load_manifest(C.MANIFEST_CSV)
    feat = pd.read_csv(os.path.join(C.ARTIFACTS_DIR, "features.csv"))
    extra = [c for c in feat.columns if c not in manifest.columns]
    df = manifest.merge(feat[["image_uid"] + extra], on="image_uid", how="left")
    df = df[~df.synthetic]

    num_cols = [c for c in extra if pd.api.types.is_numeric_dtype(df[c])]

    for name in C.VIOLATIONS:
        region = "spine" if name.startswith("spine") else "femur"
        sub = df[df.region.str.contains(region)].copy()
        y = sub["viol_" + name].values.astype(float)
        known = ~np.isnan(y)
        sub, y = sub[known], y[known].astype(int)
        groups = sub["study_uid"].values
        if len(np.unique(y)) < 2:
            continue

        cols = [c for c in st.CRITERION_FEATURES.get(name, []) if c in sub.columns]
        X = sub[cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
        base = cv_auc(X, y, groups) if cols else float("nan")

        scored = []
        for c in num_cols:
            if c not in sub.columns or c in cols:
                continue
            x = pd.to_numeric(sub[c], errors="coerce").values
            ok = ~np.isnan(x)
            if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
                continue
            a = roc_auc_score(y[ok], x[ok])
            scored.append((max(a, 1 - a), a, c))
        scored.sort(reverse=True)

        print(f"\n=== {name}  n={len(y)}  pos={int(y.sum())} ===")
        print(f"    текущий набор {cols} -> CV AUC={base:.3f}")
        print("    топ одиночных признаков (не входящих в текущий набор):")
        for s, a, c in scored[:6]:
            print(f"        {c:28} AUC={a:.3f}  (|AUC-0.5|={s:.3f})")


if __name__ == "__main__":
    main()
