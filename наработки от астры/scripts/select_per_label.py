# -*- coding: utf-8 -*-
"""Выбор источника по каждой метке: геометрия / CNN / гибрид (идея Astra №5).

CNN силён там, где геометрия слаба (укладка), и слаб там, где геометрия сильна
(ось, посторонние предметы). Единый гибрид тянет сильные критерии вниз. Здесь
источник выбирается ЧЕСТНО и ВЛОЖЕННО: внутри обучающей части внешнего фолда
считаются внутренние OOF для трёх источников, победитель по AUC применяется к
внешней валидации. Так отбор не подглядывает в оценку.

Запуск (после train):
    cd dxa_qc && python ../scripts/select_per_label.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dxa_qc"))

from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402
from src import stack as st        # noqa: E402
from src import train as tr        # noqa: E402

SOURCES = ("geo", "cnn", "fused")


def _clf():
    return Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))])


def _matrix(source, X, cnn):
    if source == "geo":
        return X
    if source == "cnn":
        return cnn.reshape(-1, 1)
    return np.column_stack([cnn, X])


def _fit_predict(source, X, cnn, y, tr, va):
    m = _clf()
    m.fit(_matrix(source, X, cnn)[tr], y[tr])
    return m.predict_proba(_matrix(source, X, cnn)[va])[:, 1]


def _nested_select(X, cnn, y, groups, n_outer=5, n_inner=4, seed=0):
    """Вложенный выбор источника: возвращает OOF-вероятности и выбранные источники."""
    oof = np.full(len(y), np.nan)
    picked = []
    outer = GroupKFold(n_splits=n_outer)
    for otr, ova in outer.split(X, y, groups=groups):
        inner_oof = {s: np.full(len(y), np.nan) for s in SOURCES}
        inner = GroupKFold(n_splits=n_inner)
        for itr, iva in inner.split(X[otr], y[otr], groups=groups[otr]):
            tr_i, va_i = otr[itr], otr[iva]
            for s in SOURCES:
                inner_oof[s][va_i] = _fit_predict(s, X, cnn, y, tr_i, va_i)
        best, best_auc = "geo", -1.0
        for s in SOURCES:
            p = inner_oof[s][otr]
            ok = ~np.isnan(p)
            if ok.sum() < 5 or len(np.unique(y[otr][ok])) < 2:
                continue
            a = roc_auc_score(y[otr][ok], p[ok])
            if a > best_auc + 1e-9:      # ничья -> геометрия (проще и устойчивее)
                best, best_auc = s, a
        picked.append(best)
        oof[ova] = _fit_predict(best, X, cnn, y, otr, ova)
    return oof, picked


def _cv_oof(source, X, cnn, y, groups, n_splits=5):
    oof = np.full(len(y), np.nan)
    for tr_i, va_i in GroupKFold(n_splits=n_splits).split(X, y, groups=groups):
        oof[va_i] = _fit_predict(source, X, cnn, y, tr_i, va_i)
    return oof


def _f1_prior(y, p):
    # ВАЖНО: это F1, а не accuracy. При 6-36 позитивах accuracy почти целиком
    # определяется негативами и к верхушке списка слепа (прежняя версия ошибочно
    # считала долю верных ответов и обесценивала сравнение источников).
    from sklearn.metrics import f1_score
    thr, _ = tr.pick_threshold(y, p, "prior")
    return float(f1_score(y, (p >= thr).astype(int), zero_division=0)), thr


def main() -> None:
    manifest = ds.load_manifest(C.MANIFEST_CSV)
    data = fl.load_features(manifest)
    real = np.asarray(ds.real_mask(data))
    oof_cnn = np.load(os.path.join(C.ARTIFACTS_DIR, "oof_violation.npy"))
    if len(oof_cnn) != len(data):
        raise SystemExit(f"OOF ({len(oof_cnn)}) не совпадает с манифестом ({len(data)})")

    print(f"{'критерий':20} {'n':>4} {'pos':>4} | {'CNN':>7} {'геом':>7} {'гибрид':>7} "
          f"{'выбор':>7} | источники по фолдам")
    acc = {s: [] for s in SOURCES}
    acc["select"] = []
    for j, name in enumerate(C.VIOLATIONS):
        region = "spine" if name.startswith("spine") else "femur"
        sub = data[data.region.str.contains(region)]
        y = sub["viol_" + name].values.astype(float)
        known = ~np.isnan(y) & real[data.region.str.contains(region).values]
        yy = y[known].astype(int)
        g = sub.loc[known, "study_uid"].values
        idx = np.where(known)[0]
        if len(np.unique(yy)) < 2:
            continue

        cols = [c for c in st.CRITERION_FEATURES.get(name, []) if c in sub.columns]
        X = sub.loc[known, cols].fillna(0).values if cols else np.zeros((len(yy), 1))
        cnn = oof_cnn[idx, j]

        per = {s: _cv_oof(s, X, cnn, yy, g) for s in SOURCES}
        # AUC по каждому источнику (по честному OOF)
        aucs = {}
        for s in SOURCES:
            ok = ~np.isnan(per[s])
            aucs[s] = roc_auc_score(yy[ok], per[s][ok])

        sel_oof, picked = _nested_select(X, cnn, yy, g)
        sel_acc, _ = _f1_prior(yy, sel_oof)
        for s in SOURCES:
            acc[s].append(_f1_prior(yy, per[s])[0])
        acc["select"].append(sel_acc)
        from collections import Counter
        print(f"{name:20} {len(yy):4d} {int(yy.sum()):4d} | {aucs['cnn']:7.3f} {aucs['geo']:7.3f} "
              f"{aucs['fused']:7.3f} {sel_acc:7.3f} | {dict(Counter(picked))}")

    print("\nmacro-F1 по критериям (усреднение):")
    for key, label in (("geo", "только геометрия"), ("cnn", "только CNN"),
                       ("fused", "гибрид (как сейчас)"), ("select", "выбор по метке (Astra №5)")):
        print(f"    {label:26} {np.mean(acc[key]):.3f}")


if __name__ == "__main__":
    main()
