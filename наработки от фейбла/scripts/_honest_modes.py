# -*- coding: utf-8 -*-
"""Прогнать честную оценку команды в разных режимах порога (prior/f1/blend).

Переиспользует eval_honest_organizer (те же источники, тот же OOF команды),
меняя только правило порога. Показывает, сколько даёт смена режима порога
на ИХ собственном стенде.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dxa_qc"))
sys.path.insert(0, HERE)

import eval_honest_organizer as E  # noqa: E402
from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402
from src import stack as st        # noqa: E402
from src.train import best_threshold, prior_threshold  # noqa: E402


def thr(y, p, mode):
    if mode == "prior":
        return prior_threshold(y, p)
    if mode == "f1":
        return best_threshold(y, p)[0]
    if mode == "blend":
        return 0.5 * (prior_threshold(y, p) + best_threshold(y, p)[0])
    raise ValueError(mode)


def honest_folds(data, real, S, source_map, mode, n_splits=5, seeds=range(10)):
    idx = np.where(real)[0]
    g = data["study_uid"].values[idx]
    per_seed = []
    for seed in seeds:
        from sklearn.model_selection import StratifiedGroupKFold
        strat = data["region"].values[idx]
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        per_fold = []
        for tr_i, va_i in splitter.split(idx, strat, groups=g):
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
                t = thr(y[trc].astype(int), p[trc], mode)
                va_ok = va[~np.isnan(p[va])]
                fired[va_ok[p[va_ok] >= t], j] = 1
            f1s = []
            for lab, crits in E.ORG_LABELS.items():
                gt = np.zeros(len(data), dtype=int)
                pr = np.zeros(len(data), dtype=int)
                for name in crits:
                    gt |= np.nan_to_num(data["viol_" + name].values.astype(float),
                                        nan=0.0).astype(int)
                    pr |= fired[:, C.VIOLATION_IDX[name]]
                f1s.append(f1_score(gt[va], pr[va], zero_division=0))
            per_fold.append(float(np.mean(f1s)))
        per_seed.append(np.mean(per_fold))
    return float(np.mean(per_seed)), float(np.std(per_seed))


def main() -> int:
    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    S = E.build_scores(data, real)
    crits = list(C.VIOLATIONS)
    configs = {
        "fused везде": {c: "fused" for c in crits},
        "текущая (config)": {c: st.source_of(c) for c in crits},
    }
    print("честная оценка на OOF команды, 10 разбиений:")
    for title, sm in configs.items():
        for mode in ("prior", "f1", "blend"):
            m, s = honest_folds(data, real, S, sm, mode)
            print(f"  {title:18} {mode:6}  macro-F1={m:.3f}±{s:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
