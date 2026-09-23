# -*- coding: utf-8 -*-
"""Стенд: метрика организатора для любой комбинации источников по критериям.

В режиме ``prior`` порог ставится по доле нарушений, поэтому решение целиком
определяется РАНЖИРОВАНИЕМ в top-k, а не глобальным AUC. Здесь для каждого
критерия можно выбрать источник скора (CNN / геометрия / гибрид) и посчитать
итоговые метрики организатора:

  * macro-F1 по 4 уникальным меткам (главная);
  * метрики quality_class (balanced accuracy / macro-F1 / ROC-AUC).

Источники:
  cnn   — OOF-вероятность нейросети (oof_violation.npy);
  fused — гибридная шкала стекера (stack_oof_violation.npy);
  geo   — логистическая регрессия по геометрическим признакам, честный OOF-CV.

Запуск (после train + stack):
    cd dxa_qc && python ../scripts/eval_source_configs.py
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
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
    "Некорректная укладка": ["spine_positioning", "femur_positioning"],
    "Не выравнена ось позвоночника": ["spine_axis"],
    "Присутствуют посторонние предметы": ["spine_artifacts"],
    "Некорректная область интереса": ["femur_roi"],
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


def build_scores(data, real):
    """Скоры [n, 5] по всем трём источникам + маска реальных строк."""
    n = len(data)
    oof_cnn = np.load(os.path.join(C.ARTIFACTS_DIR, "oof_violation.npy"))
    fused = np.load(os.path.join(C.ARTIFACTS_DIR, "stack_oof_violation.npy"))
    S = {s: np.full((n, len(C.VIOLATIONS)), np.nan) for s in SOURCES}
    for j, name in enumerate(C.VIOLATIONS):
        key = "spine" if name.startswith("spine") else "femur"
        rmask = data["region"].str.contains(key).values & real
        y = data["viol_" + name].values.astype(float)
        known = ~np.isnan(y) & rmask
        if known.sum() == 0:
            continue
        S["cnn"][known, j] = oof_cnn[known, j]
        S["fused"][known, j] = fused[known, j]
        cols = [c for c in st.CRITERION_FEATURES.get(name, []) if c in data.columns]
        X = data.loc[known, cols].fillna(0).values if cols else np.zeros((known.sum(), 1))
        g = data.loc[known, "study_uid"].values
        S["geo"][known, j] = _cv_geo(X, y[known].astype(int), g)
    return S


def decide(data, real, S, source_map):
    """Сработавшие критерии по порогу prior; возвращает fired [n,5] и quality [n]."""
    n = len(data)
    fired = np.zeros((n, len(C.VIOLATIONS)), dtype=int)
    for j, name in enumerate(C.VIOLATIONS):
        key = "spine" if name.startswith("spine") else "femur"
        rmask = data["region"].str.contains(key).values & real
        y = data["viol_" + name].values.astype(float)
        known = ~np.isnan(y) & rmask
        p = S[source_map[name]][:, j]
        ok = known & ~np.isnan(p)
        if ok.sum() < 2 or len(np.unique(y[ok].astype(int))) < 2:
            continue
        thr = prior_threshold(y[ok].astype(int), p[ok])
        fired[ok & (p >= thr), j] = 1
    quality = (fired.sum(axis=1) > 0).astype(int)
    return fired, quality


def score(data, real, S, source_map, verbose=False):
    fired, quality = decide(data, real, S, source_map)
    f1s, per = [], {}
    for label, crits in ORG_LABELS.items():
        gt = np.zeros(len(data), dtype=int)
        pr = np.zeros(len(data), dtype=int)
        for name in crits:
            col = "viol_" + name
            v = data[col].values.astype(float)
            gt |= np.nan_to_num(v, nan=0.0).astype(int)
            pr |= fired[:, C.VIOLATION_IDX[name]]
        f1 = f1_score(gt[real], pr[real], zero_division=0)
        f1s.append(f1)
        per[label] = f1
    yq = data["quality"].values.astype(float)
    known = ~np.isnan(yq) & real
    yqb = np.nan_to_num(yq, nan=0.0).astype(int)[known]
    pq = quality[known]
    ba = balanced_accuracy_score(yqb, pq)
    mf1 = f1_score(yqb, pq, average="macro", zero_division=0)
    auc = roc_auc_score(yqb, pq) if len(np.unique(yqb)) > 1 else float("nan")
    if verbose:
        for label, f in per.items():
            print(f"    {label:34} F1={f:.3f}")
        print(f"    quality: BA={ba:.3f} macroF1={mf1:.3f} AUC={auc:.3f}")
    return float(np.mean(f1s)), dict(per=per, ba=ba, mf1=mf1, auc=auc, macro=float(np.mean(f1s)))


def main() -> None:
    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    S = build_scores(data, real)
    crits = list(C.VIOLATIONS)

    print("=== Источники по отдельности (все критерии одним источником) ===")
    for s in SOURCES:
        m, _ = score(data, real, S, {c: s for c in crits})
        print(f"  все={s:6} macro-F1(4 метки) = {m:.3f}")

    print("\n=== Текущий пайплайн (гибрид везде) подробно ===")
    score(data, real, S, {c: "fused" for c in crits}, verbose=True)

    print("\n=== Полный перебор 3^5 = 243 комбинаций ===")
    rows = []
    for combo in itertools.product(SOURCES, repeat=len(crits)):
        sm = dict(zip(crits, combo))
        m, d = score(data, real, S, sm)
        rows.append((m, combo, d))
    rows.sort(key=lambda r: -r[0])
    print(f"{'macro-F1':>9}  комбинация (spine_pos, spine_axis, spine_art, femur_pos, femur_roi)")
    for m, combo, d in rows[:12]:
        print(f"{m:9.3f}  {combo}")
    print("  ...")
    for m, combo, d in rows[-3:]:
        print(f"{m:9.3f}  {combo}")
    best_m, best_combo, best_d = rows[0]
    print(f"\nЛучшее: macro-F1={best_m:.3f}  {dict(zip(crits, best_combo))}")
    print(f"  quality: BA={best_d['ba']:.3f} macroF1={best_d['mf1']:.3f} AUC={best_d['auc']:.3f}")

    # «доменно-осмысленное» правило: CNN везде, кроме оси позвоночника (там CNN AUC 0.38)
    sm = {"spine_positioning": "cnn", "spine_axis": "fused", "spine_artifacts": "cnn",
          "femur_positioning": "cnn", "femur_roi": "cnn"}
    m, d = score(data, real, S, sm)
    print(f"\n=== Правило: CNN везде, fused только для spine_axis ===")
    print(f"  macro-F1={m:.3f}  BA={d['ba']:.3f} macroF1={d['mf1']:.3f} AUC={d['auc']:.3f}")
    for label, f in d["per"].items():
        print(f"    {label:34} F1={f:.3f}")


if __name__ == "__main__":
    main()
