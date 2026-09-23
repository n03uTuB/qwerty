# -*- coding: utf-8 -*-
"""Честный F1 по КАЖДОМУ критерию ТЗ (а не по 4 меткам организатора).

Помогает понять, внутри какой метки сидит слабое место: «укладка» объединяет
spine_positioning и femur_positioning, поэтому её F1 не говорит, какой из двух
критериев тянет вниз. Порог — по train-части фолда, предсказания объединяются
(как один отложенный набор), 20 разбиений.

Запуск:
    cd dxa_qc && python ../scripts/diag_final_per_criterion.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
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
    out = {}
    for j, name in enumerate(C.VIOLATIONS):
        key = "spine" if name.startswith("spine") else "femur"
        rm = data["region"].str.contains(key).values & real
        y = data["viol_" + name].values.astype(float)
        kn = rm & ~np.isnan(y)
        yy = y[kn].astype(int)
        pp = fired[kn, j]
        p_cont = S[sm[name]][kn, j]
        out[name] = dict(
            f1=f1_score(yy, pp, zero_division=0),
            pos=int(yy.sum()), n=int(len(yy)),
            pred=int(pp.sum()),
            auc=roc_auc_score(yy, p_cont) if len(np.unique(yy)) > 1 else float("nan"),
            ap=average_precision_score(yy, p_cont) if len(np.unique(yy)) > 1 else float("nan"),
        )
    return out


def main() -> None:
    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    S = E.build_scores(data, real)
    sm = {c: st.source_of(c) for c in C.VIOLATIONS}
    print(f"Конфигурация: {sm}, режим порога: {C.THRESHOLD_MODE}")
    print(f"Честный F1 по критериям, {N_SEEDS} разбиений:\n")
    res = [run(data, real, S, sm, s) for s in range(N_SEEDS)]
    print(f"{'критерий':20} {'источник':9} {'F1':>7} {'±':>6} {'pos':>4} {'pred':>5} {'AUC':>6} {'AP':>6}")
    for name in C.VIOLATIONS:
        f1s = np.array([r[name]["f1"] for r in res])
        aucs = np.array([r[name]["auc"] for r in res])
        aps = np.array([r[name]["ap"] for r in res])
        r0 = res[0][name]
        print(f"{name:20} {sm[name]:9} {f1s.mean():7.3f} {f1s.std():6.3f} "
              f"{r0['pos']:4d} {r0['pred']:5d} {aucs.mean():6.3f} {aps.mean():6.3f}")


if __name__ == "__main__":
    main()
