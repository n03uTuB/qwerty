# -*- coding: utf-8 -*-
"""Диагностика признаков: одномерные AUC и жадный отбор по критериям (быстро, без CV-петли).

Для каждого критерия считает AUC каждого признака и жадный forward-отбор по AUC
(логистика, GroupKFold). Помогает найти лучший набор для слабых меток.

Запуск:
    python dxa_qc_work/scripts/exp_feat_auc.py
"""
from __future__ import annotations

import os

import numpy as np

import _common  # noqa: F401

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402

FEATS = [f for f in fl.FEATURE_NAMES if f != "copies"]


def cv_auc(data, real, name, cols):
    j = C.VIOLATION_IDX[name]
    reg = "spine" if name.startswith("spine") else "femur"
    m = data["region"].str.contains(reg).values & real
    y = data["viol_" + name].values.astype(float)
    known = ~np.isnan(y) & m
    X = data.loc[known, cols].fillna(0).values
    yk = y[known].astype(int)
    gk = data.loc[known, "study_uid"].values
    if len(np.unique(yk)) < 2:
        return float("nan")
    p = np.full(len(yk), np.nan)
    for tr, va in GroupKFold(n_splits=5).split(X, yk, groups=gk):
        if len(np.unique(yk[tr])) < 2:
            continue
        clf = Pipeline([("s", StandardScaler()),
                        ("c", LogisticRegression(max_iter=2000, class_weight="balanced"))])
        p[va] = clf.fit(X[tr], yk[tr]).predict_proba(X[va])[:, 1]
    ok = ~np.isnan(p)
    return float(roc_auc_score(yk[ok], p[ok])) if len(np.unique(yk[ok])) > 1 else float("nan")


def main():
    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))

    for name in C.VIOLATIONS:
        reg = "spine" if name.startswith("spine") else "femur"
        y = data["viol_" + name].values.astype(float)
        m = data["region"].str.contains(reg).values & real & ~np.isnan(y)
        pos = int((y[m] > 0.5).sum())
        print(f"\n=== {name}  (n={int(m.sum())}, pos={pos}) ===")
        # одномерные AUC
        uni = []
        for f in FEATS:
            x = data.loc[m, f].values.astype(float)
            ok = ~np.isnan(x)
            if ok.sum() > 10 and len(np.unique(y[m][ok])) > 1:
                a = roc_auc_score(y[m][ok].astype(int), x[ok])
                uni.append((max(a, 1 - a), a, f))
        uni.sort(reverse=True)
        print("  топ одномерных:", "  ".join(f"{f}={a:.3f}" for _, a, f in uni[:6]))
        # жадный отбор
        chosen, best = [], 0.0
        for _ in range(6):
            cand_best, cand_f = best, None
            for f in FEATS:
                if f in chosen:
                    continue
                a = cv_auc(data, real, name, chosen + [f])
                if not np.isnan(a) and a > cand_best + 1e-4:
                    cand_best, cand_f = a, f
            if cand_f is None:
                break
            chosen.append(cand_f)
            best = cand_best
            print(f"  +{cand_f:24} AUC={best:.3f}")
        print(f"  ИТОГ: {chosen}  AUC={best:.3f}")


if __name__ == "__main__":
    main()
