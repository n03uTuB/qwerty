# -*- coding: utf-8 -*-
"""Честная проверка «сырых» одиночных признаков как источника по критерию.

Идея: часть критериев ТЗ — прямое измерение (ось <=5 град, отступ поля). Иногда
один семантически прямой признак (без обученной логистики) ранжирует верхушку
списка не хуже, а то и лучше стекера. Проверяем это ЧЕСТНО: порог prior считается
по train-части фолда, метрика — по val, на 10 разбиениях.

Запуск:
    cd dxa_qc && python ../scripts/eval_raw_features_honest.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dxa_qc"))

from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402
from src import stack as st        # noqa: E402

import eval_honest_organizer as E  # noqa: E402

N_SEEDS = 10


def macro_f1(data, real, scores, source_map, seed):
    """scores: dict источник -> [n,5] матрица. source_map: критерий -> источник."""
    idx = np.where(real)[0]
    g = data["study_uid"].values[idx]
    strat = data["region_idx"].values[idx] if "region_idx" in data.columns else None
    if strat is None or len(np.unique(strat)) < 2:
        from sklearn.model_selection import GroupKFold
        splits = GroupKFold(n_splits=5).split(idx, groups=g)
    else:
        splits = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                      random_state=seed).split(idx, strat, groups=g)
    vals = []
    for tr_i, va_i in splits:
        tr, va = idx[tr_i], idx[va_i]
        fired = np.zeros((len(data), len(C.VIOLATIONS)), dtype=int)
        for j, name in enumerate(C.VIOLATIONS):
            key = "spine" if name.startswith("spine") else "femur"
            rmask = data["region"].str.contains(key).values
            y = data["viol_" + name].values.astype(float)
            p = scores[source_map[name]][:, j]
            trc = tr[rmask[tr] & ~np.isnan(y[tr]) & ~np.isnan(p[tr])]
            if len(trc) < 2 or len(np.unique(y[trc].astype(int))) < 2:
                continue
            thr = E.prior_threshold(y[trc].astype(int), p[trc])
            va_ok = va[~np.isnan(p[va])]
            fired[va_ok[p[va_ok] >= thr], j] = 1
        fs = []
        for crits in E.ORG_LABELS.values():
            gt = np.zeros(len(data), dtype=int)
            pr = np.zeros(len(data), dtype=int)
            for name in crits:
                v = data["viol_" + name].values.astype(float)
                gt |= np.nan_to_num(v, nan=0.0).astype(int)
                pr |= fired[:, C.VIOLATION_IDX[name]]
            fs.append(E.f1_score(gt[va], pr[va], zero_division=0))
        vals.append(float(np.mean(fs)))
    return float(np.mean(vals))


def main() -> None:
    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    S = E.build_scores(data, real)
    crits = list(C.VIOLATIONS)
    base = {c: st.source_of(c) for c in crits}

    def ev(sm):
        return float(np.mean([macro_f1(data, real, S, sm, s) for s in range(N_SEEDS)]))

    base_score = ev(base)
    print(f"Базовая конфигурация (config): {base_score:.3f}")
    print(f"  {base}\n")

    print("Перебор «сырых» признаков (один критерий заменяется на ±признак):")
    results = []
    for c in crits:
        key = "spine" if c.startswith("spine") else "femur"
        cols = [x for x in data.columns if x.startswith(key + "_")]
        best = None
        for col in cols:
            v = data[col].values.astype(float)
            if np.nanstd(v) == 0:
                continue
            for sign, tag in ((1.0, "+"), (-1.0, "-")):
                src = f"raw_{col}_{tag}"
                S[src] = np.full((len(data), len(C.VIOLATIONS)), np.nan)
                S[src][:, C.VIOLATION_IDX[c]] = sign * v
                sm = dict(base)
                sm[c] = src
                m = ev(sm)
                if best is None or m > best[0]:
                    best = (m, col, tag)
        results.append((best[0] - base_score, c, best))
        print(f"  {c:20} лучший: {best[2]}{best[1]:26} "
              f"macro-F1={best[0]:.3f}  ({best[0]-base_score:+.3f})")

    print("\nЕсли применить все лучшие «сырые» замены одновременно:")
    sm = dict(base)
    for _, c, best in results:
        src = f"best_{c}"
        S[src] = np.full((len(data), len(C.VIOLATIONS)), np.nan)
        sign = 1.0 if best[2] == "+" else -1.0
        S[src][:, C.VIOLATION_IDX[c]] = sign * data[best[1]].values.astype(float)
        sm[c] = src
    print(f"  macro-F1={ev(sm):.3f}  (база {base_score:.3f})")


if __name__ == "__main__":
    main()
