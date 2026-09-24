# -*- coding: utf-8 -*-
"""Проверка устойчивости выбора источника по критериям (не переобучение ли).

Выбор «CNN везде, гибрид только для оси» сделан по полным OOF. Здесь проверяем
честно: внутри каждого из 5 внешних фолдов порог prior считается по обучающей
части, метрика — по валидационной. Смотрим, повторяется ли выбор источника от
фолда к фолду (если «прыгает» — выбор был случайным).

Запуск:
    cd dxa_qc && python ../scripts/check_source_stability.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dxa_qc"))

from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402
from src import stack as st        # noqa: E402
from src.train import prior_threshold  # noqa: E402

SOURCES = ("cnn", "geo", "fused")


def _clf():
    return Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(max_iter=2000,
                                              class_weight="balanced", C=1.0))])


def _cv_geo(X, y, groups):
    oof = np.full(len(y), np.nan)
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups=groups):
        if len(np.unique(y[tr])) < 2:
            continue
        m = _clf()
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    return oof


def _f1(y, p, tr, va):
    thr = prior_threshold(y[tr], p[tr])
    return float(f1_score(y[va], (p[va] >= thr).astype(int), zero_division=0))


def main() -> None:
    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    oof_cnn = np.load(os.path.join(C.ARTIFACTS_DIR, "oof_violation.npy"))
    fused = np.load(os.path.join(C.ARTIFACTS_DIR, "stack_oof_violation.npy"))

    print(f"{'критерий':20} {'источник':9} " + " ".join(f"ф{i}" for i in range(5))
          + "   средн  победители по фолдам")
    overall = {s: [] for s in SOURCES}
    for j, name in enumerate(C.VIOLATIONS):
        key = "spine" if name.startswith("spine") else "femur"
        rmask = data["region"].str.contains(key).values & real
        y = data["viol_" + name].values.astype(float)
        known = ~np.isnan(y) & rmask
        yy = y[known].astype(int)
        g = data.loc[known, "study_uid"].values
        cols = [c for c in st.CRITERION_FEATURES.get(name, []) if c in data.columns]
        X = data.loc[known, cols].fillna(0).values if cols else np.zeros((len(yy), 1))
        scores = {"cnn": oof_cnn[known, j], "fused": fused[known, j],
                  "geo": _cv_geo(X, yy, g)}
        folds = list(GroupKFold(n_splits=5).split(X, yy, groups=g))
        per_fold = {}
        for s in SOURCES:
            f1s = []
            for tr, va in folds:
                if len(np.unique(yy[tr])) < 2:
                    f1s.append(float("nan"))
                    continue
                f1s.append(_f1(yy, scores[s], tr, va))
            per_fold[s] = f1s
            overall[s].append(np.nanmean(f1s))
        winners = []
        for fi in range(5):
            best = max(SOURCES, key=lambda s: (per_fold[s][fi]
                                               if not np.isnan(per_fold[s][fi]) else -1))
            winners.append(best)
        for s in SOURCES:
            f1s = per_fold[s]
            cells = " ".join("  -  " if np.isnan(v) else f"{v:.2f}" for v in f1s)
            tail = dict(Counter(winners)) if s == SOURCES[0] else ""
            print(f"{name if s == SOURCES[0] else '':20} {s:9} {cells}   "
                  f"{np.nanmean(f1s):.3f}  {tail}")
    print()
    print("средний F1@prior по критериям:")
    for s in SOURCES:
        print(f"  {s:6} {np.mean(overall[s]):.3f}")


if __name__ == "__main__":
    main()
