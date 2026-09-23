# -*- coding: utf-8 -*-
"""Порог prior со «запасом»: k = alpha * доля * n. Подбор alpha честно.

Наблюдение: в честной CV режим prior систематически НЕДОпредсказывает долю
нарушений на валидации (например, для оси: 0.05-0.10 против истинных 0.15-0.16),
из-за чего теряется recall и падает F1. Режим f1 берёт более низкий порог и
выигрывает (0.415 против 0.358), но его порог — прямой оптимум F1 на train, что
менее устойчиво.

Середина: оставить устойчивость оценки по доле, но ввести коэффициент запаса
alpha >= 1: k = max(1, round(alpha * доля * n)). alpha подбирается ЧЕСТНО
(порог по train-части, метрика по val, 20 разбиений).

Запуск:
    cd dxa_qc && python ../scripts/tune_prior_alpha.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dxa_qc"))

from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402
from src import stack as st        # noqa: E402
from src.train import best_threshold  # noqa: E402

import eval_honest_organizer as E  # noqa: E402

N_SEEDS = 20


def alpha_threshold(y, p, alpha):
    prev = float(np.mean(y)) if len(y) else 0.0
    if prev <= 0 or prev >= 1 or len(p) == 0:
        return 0.5
    count = max(1, int(round(alpha * prev * len(p))))
    count = min(count, len(p))
    return float(np.sort(p)[::-1][count - 1])


def macro_f1(data, real, S, source_map, seed, alpha=None):
    idx = np.where(real)[0]
    g = data["study_uid"].values[idx]
    strat = data["region_idx"].values[idx] if "region_idx" in data.columns else None
    if strat is None or len(np.unique(strat)) < 2:
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
            p = S[source_map[name]][:, j]
            trc = tr[rmask[tr] & ~np.isnan(y[tr]) & ~np.isnan(p[tr])]
            if len(trc) < 2 or len(np.unique(y[trc].astype(int))) < 2:
                continue
            yt, pt = y[trc].astype(int), p[trc]
            thr = best_threshold(yt, pt)[0] if alpha is None else alpha_threshold(yt, pt, alpha)
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
    sm = {c: st.source_of(c) for c in C.VIOLATIONS}
    print(f"Конфигурация: {sm}")
    print(f"Честная macro-F1 по 4 меткам, {N_SEEDS} разбиений:\n")
    print(f"{'порог':22} {'средн':>7} {'медиана':>8} {'ст.откл':>8}  мин..макс")
    for alpha in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0):
        vals = np.array([macro_f1(data, real, S, sm, s, alpha) for s in range(N_SEEDS)])
        print(f"{'prior alpha=%.2f' % alpha:22} {vals.mean():7.3f} {np.median(vals):8.3f} "
              f"{vals.std():8.3f}  {vals.min():.2f}..{vals.max():.2f}")
    vals = np.array([macro_f1(data, real, S, sm, s, None) for s in range(N_SEEDS)])
    print(f"{'f1 (оптимум на train)':22} {vals.mean():7.3f} {np.median(vals):8.3f} "
          f"{vals.std():8.3f}  {vals.min():.2f}..{vals.max():.2f}")


if __name__ == "__main__":
    main()
