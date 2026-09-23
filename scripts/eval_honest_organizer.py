# -*- coding: utf-8 -*-
"""ЧЕСТНАЯ оценка метрики организатора: порог по train-части фолда, метрика — по val.

Зачем отдельный скрипт. ``thresholds.json`` подбирается по тем же OOF-предсказаниям,
по которым считается ``report.md``, поэтому цифры в отчёте оптимистичны. Организатор
же оценит на отложенной выборке. Здесь пороги (режим prior) считаются ВНУТРИ
обучающей части каждого внешнего фолда, а метрика — на валидационной. Это оценка
того, что реально покажет пайплайн.

Сравниваются три конфигурации источников:
  * fused везде (как было до правки);
  * cnn везде;
  * текущая из config.CRITERION_SOURCES (CNN везде, гибрид только для оси).

Запуск:
    cd dxa_qc && python ../scripts/eval_honest_organizer.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
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

ORG_LABELS = {
    "укладка": ["spine_positioning", "femur_positioning"],
    "ось": ["spine_axis"],
    "предметы": ["spine_artifacts"],
    "ROI": ["femur_roi"],
}
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


def _cv_fuse(X, cnn, y, groups):
    """Гибридный скор (CNN + геометрия) с кросс-валидацией по study_uid."""
    Xf = np.column_stack([cnn, X])
    oof = np.full(len(y), np.nan)
    for tr, va in GroupKFold(n_splits=5).split(Xf, y, groups=groups):
        if len(np.unique(y[tr])) < 2:
            continue
        m = _clf()
        m.fit(Xf[tr], y[tr])
        oof[va] = m.predict_proba(Xf[va])[:, 1]
    return oof


def build_scores(data, real):
    n = len(data)
    oof_cnn = np.load(os.path.join(C.ARTIFACTS_DIR, "oof_violation.npy"))
    fused = np.load(os.path.join(C.ARTIFACTS_DIR, "stack_oof_violation.npy"))
    S = {s: np.full((n, len(C.VIOLATIONS)), np.nan)
         for s in SOURCES + ("fusedold",)}
    for j, name in enumerate(C.VIOLATIONS):
        key = "spine" if name.startswith("spine") else "femur"
        rmask = data["region"].str.contains(key).values & real
        y = data["viol_" + name].values.astype(float)
        known = ~np.isnan(y) & rmask
        S["cnn"][known, j] = oof_cnn[known, j]
        S["fused"][known, j] = fused[known, j]
        cols = [c for c in st.CRITERION_FEATURES.get(name, []) if c in data.columns]
        X = data.loc[known, cols].fillna(0).values if cols else np.zeros((known.sum(), 1))
        yk = y[known].astype(int)
        gk = data.loc[known, "study_uid"].values
        S["geo"][known, j] = _cv_geo(X, yk, gk)
        # старое поведение: гибрид для ВСЕХ критериев (даже там, где CNN сильнее)
        S["fusedold"][known, j] = _cv_fuse(X, oof_cnn[known, j], yk, gk)
    return S


def honest_folds(data, real, S, source_map, n_splits=5):
    """F1 по 4 меткам организатора на каждом фолде (порог prior — только по train)."""
    idx = np.where(real)[0]
    g = data["study_uid"].values[idx]
    groups = data["region"].values
    per_fold, per_label_fold = [], {k: [] for k in ORG_LABELS}
    for tr_i, va_i in GroupKFold(n_splits=n_splits).split(idx, groups=g):
        tr, va = idx[tr_i], idx[va_i]
        fired = np.zeros((len(data), len(C.VIOLATIONS)), dtype=int)
        for j, name in enumerate(C.VIOLATIONS):
            key = "spine" if name.startswith("spine") else "femur"
            rmask = data["region"].str.contains(key).values
            y = data["viol_" + name].values.astype(float)
            p = S[source_map[name]][:, j]
            trc = tr[rmask[tr] & ~np.isnan(y[tr]) & ~np.isnan(p[tr])]
            if len(trc) < 2 or len(np.unique(y[trc].astype(int))) < 2:
                continue
            thr = prior_threshold(y[trc].astype(int), p[trc])
            va_ok = va[~np.isnan(p[va])]
            pred = p[va_ok] >= thr
            fired[va_ok[pred], j] = 1
        f1s = []
        for lab, crits in ORG_LABELS.items():
            gt = np.zeros(len(data), dtype=int)
            pr = np.zeros(len(data), dtype=int)
            for name in crits:
                v = data["viol_" + name].values.astype(float)
                gt |= np.nan_to_num(v, nan=0.0).astype(int)
                pr |= fired[:, C.VIOLATION_IDX[name]]
            f1s.append(f1_score(gt[va], pr[va], zero_division=0))
            per_label_fold[lab].append(f1_score(gt[va], pr[va], zero_division=0))
        per_fold.append(float(np.mean(f1s)))
    return per_fold, per_label_fold


def main() -> None:
    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    S = build_scores(data, real)
    crits = list(C.VIOLATIONS)

    configs = {
        "fused везде (как было)": {c: "fusedold" for c in crits},
        "cnn везде": {c: "cnn" for c in crits},
        "геометрия только для оси": {c: ("geo" if c == "spine_axis" else "cnn")
                                     for c in crits},
        "текущая (config)": {c: st.source_of(c) for c in crits},
    }
    print("Честная оценка: порог prior считается по train-части фолда.\n")
    print(f"{'конфигурация':24} {'macro-F1(4 метки)':>18}   по фолдам")
    for title, sm in configs.items():
        pf, plf = honest_folds(data, real, S, sm)
        folds = " ".join(f"{v:.2f}" for v in pf)
        print(f"{title:24} {np.mean(pf):18.3f}   {folds}")
        print(f"{'':24} " + "  ".join(f"{k}={np.mean(v):.3f}" for k, v in plf.items()))


if __name__ == "__main__":
    main()
