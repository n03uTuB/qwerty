# -*- coding: utf-8 -*-
"""Повторная честная оценка конфигураций источников (устойчивость к разбиению).

Разница между «геометрия для оси» и «гибрид для оси» на одном 5-фолдовом
разбиении (0.408 против 0.386) лежит в пределах шума: при 10 позитивах ДИ на F1
широкий. Здесь одно и то же сравнивается на 30 разных разбиениях
(StratifiedGroupKFold по study_uid), и приводятся среднее и разброс.

Запуск:
    cd dxa_qc && python ../scripts/compare_source_repeated.py
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

import eval_honest_organizer as E  # noqa: E402  (переиспользуем сборку скоров)

N_SEEDS = 30


def honest_macro_f1(data, real, S, source_map, seed):
    idx = np.where(real)[0]
    g = data["study_uid"].values[idx]
    strat = data["region_idx"].values[idx] if "region_idx" in data.columns else None
    if strat is None or len(np.unique(strat)) < 2:
        from sklearn.model_selection import GroupKFold
        splitter = GroupKFold(n_splits=5)
        splits = splitter.split(idx, groups=g)
    else:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        splits = splitter.split(idx, strat, groups=g)
    f1s_all = []
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
        f1s_all.append(float(np.mean(fs)))
    return float(np.mean(f1s_all))


def main() -> None:
    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    S = E.build_scores(data, real)
    crits = list(C.VIOLATIONS)
    configs = {
        "гибрид везде (как было)": {c: "fusedold" for c in crits},
        "CNN везде": {c: "cnn" for c in crits},
        "CNN, геометрия для оси": {c: ("geo" if c == "spine_axis" else "cnn")
                                   for c in crits},
        "CNN, гибрид для оси (сейчас)": {c: ("fused" if c == "spine_axis" else "cnn")
                                         for c in crits},
    }
    print(f"Честная macro-F1 по 4 меткам, {N_SEEDS} разбиений "
          f"(StratifiedGroupKFold по study_uid):\n")
    print(f"{'конфигурация':32} {'средн':>7} {'медиана':>8} {'ст.откл':>8}  мин..макс")
    for title, sm in configs.items():
        vals = [honest_macro_f1(data, real, S, sm, s) for s in range(N_SEEDS)]
        vals = np.array(vals)
        print(f"{title:32} {vals.mean():7.3f} {np.median(vals):8.3f} "
              f"{vals.std():8.3f}  {vals.min():.2f}..{vals.max():.2f}")


if __name__ == "__main__":
    main()
