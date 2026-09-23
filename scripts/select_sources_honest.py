# -*- coding: utf-8 -*-
"""Выбор источника по критерию ЗАНОВО — в честной схеме и в режиме f1.

Предыдущий перебор (eval_source_configs.py) был по полному OOF и в режиме prior.
Здесь то же самое, но честно: порог по train-части фолда (режим config.THRESHOLD_MODE),
метрика по объединённым val-предсказаниям, 20 разбиений. Каждый критерий
оптимизируется независимо; spine_positioning и femur_positioning образуют общую
метку «укладка», поэтому для них считается и совместный вариант.

Запуск:
    cd dxa_qc && python ../scripts/select_sources_honest.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dxa_qc"))

from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402
from src import stack as st        # noqa: E402
from src.train import pick_threshold  # noqa: E402

import eval_honest_organizer as E  # noqa: E402

N_SEEDS = 20
SOURCES = ("cnn", "geo", "fused")


def run(data, real, S, sm, seed):
    idx = np.where(real)[0]
    g = data["study_uid"].values[idx]
    strat = data["region_idx"].values[idx] if "region_idx" in data.columns else None
    if strat is None or len(np.unique(strat)) < 2:
        splits = GroupKFold(n_splits=5).split(idx, groups=g)
    else:
        splits = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                      random_state=seed).split(idx, strat, groups=g)
    fired = np.zeros((len(data), len(C.VIOLATIONS)), dtype=int)
    for tr_i, va_i in splits:
        tr, va = idx[tr_i], idx[va_i]
        for j, name in enumerate(C.VIOLATIONS):
            key = "spine" if name.startswith("spine") else "femur"
            rmask = data["region"].str.contains(key).values
            y = data["viol_" + name].values.astype(float)
            p = S[sm[name]][:, j]
            trc = tr[rmask[tr] & ~np.isnan(y[tr]) & ~np.isnan(p[tr])]
            if len(trc) < 2 or len(np.unique(y[trc].astype(int))) < 2:
                continue
            thr, _ = pick_threshold(y[trc].astype(int), p[trc], C.THRESHOLD_MODE)
            va_ok = va[~np.isnan(p[va])]
            fired[va_ok[p[va_ok] >= thr], j] = 1
    per, fs = {}, []
    for lab, crits in E.ORG_LABELS.items():
        gt = np.zeros(len(data), dtype=int)
        pr = np.zeros(len(data), dtype=int)
        for name in crits:
            v = data["viol_" + name].values.astype(float)
            gt |= np.nan_to_num(v, nan=0.0).astype(int)
            pr |= fired[:, C.VIOLATION_IDX[name]]
        f = f1_score(gt[idx], pr[idx], zero_division=0)
        per[lab] = f
        fs.append(f)
    return float(np.mean(fs)), per


def ev(data, real, S, sm):
    res = [run(data, real, S, sm, s) for s in range(N_SEEDS)]
    return (float(np.mean([r[0] for r in res])),
            {k: float(np.mean([r[1][k] for r in res])) for k in E.ORG_LABELS})


def main() -> None:
    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    S = E.build_scores(data, real)
    base = {c: st.source_of(c) for c in C.VIOLATIONS}
    m0, p0 = ev(data, real, S, base)
    print(f"База ({C.THRESHOLD_MODE}): macro-F1 {m0:.3f}  "
          + "  ".join(f"{k}={v:.3f}" for k, v in p0.items()))
    print(f"  {base}\n")

    print("Перебор источника по одному критерию (остальные — из базы):")
    best = dict(base)
    for c in C.VIOLATIONS:
        row = []
        for s in SOURCES:
            sm = dict(best)
            sm[c] = s
            m, p = ev(data, real, S, sm)
            row.append((m, s, p))
        row.sort(reverse=True)
        print(f"  {c:20} " + "  ".join(f"{s}={m:.3f}" for m, s, _ in
                                       sorted(row, key=lambda r: SOURCES.index(r[1]))))
        best[c] = row[0][1]
    m1, p1 = ev(data, real, S, best)
    print(f"\nЛучшее по одному критерию: macro-F1 {m1:.3f}  "
          + "  ".join(f"{k}={v:.3f}" for k, v in p1.items()))
    print(f"  {best}")


if __name__ == "__main__":
    main()
