# -*- coding: utf-8 -*-
"""Финальная ЧЕСТНАЯ оценка развёрнутой конфигурации пайплайна.

Развёрнутая конфигурация: источники из config.CRITERION_SOURCES, режим порога
config.THRESHOLD_MODE. Здесь она оценивается честно — порог подбирается только
по train-части фолда, метрика считается на val, затем результаты объединяются
(как если бы организатор оценил один отложенный набор). 20 разбиений
StratifiedGroupKFold по study_uid.

Метрики: macro-F1 по 4 меткам организатора, balanced accuracy / macro-F1 / ROC-AUC
для quality_class (вероятность качества = max по критериям области).

Запуск:
    cd dxa_qc && python ../scripts/eval_final.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
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


def run(data, real, S, sm, seed, mode=None):
    mode = mode or C.THRESHOLD_MODE
    idx = np.where(real)[0]
    g = data["study_uid"].values[idx]
    strat = data["region_idx"].values[idx] if "region_idx" in data.columns else None
    if strat is None or len(np.unique(strat)) < 2:
        splits = GroupKFold(n_splits=5).split(idx, groups=g)
    else:
        splits = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                      random_state=seed).split(idx, strat, groups=g)
    fired = np.zeros((len(data), len(C.VIOLATIONS)), dtype=int)
    qprob = np.full(len(data), np.nan)
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
            thr, _ = pick_threshold(y[trc].astype(int), p[trc], mode)
            va_ok = va[~np.isnan(p[va])]
            fired[va_ok[p[va_ok] >= thr], j] = 1
        for i in va:
            crit = C.REGION_CRITERIA[data["region"].values[i]]
            vals = [S[sm[n]][i, C.VIOLATION_IDX[n]] for n in crit]
            vals = [float(v) for v in vals if not np.isnan(v)]
            if vals:
                qprob[i] = max(vals)

    fs, per = [], {}
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
    yq = data["quality"].values.astype(float)
    kn = ~np.isnan(yq) & real
    yqb = np.nan_to_num(yq, nan=0.0).astype(int)[kn]
    pq = (fired.sum(axis=1) > 0).astype(int)[kn]
    prq = qprob[kn]
    ok = ~np.isnan(prq)
    return dict(
        macro=float(np.mean(fs)), per=per,
        ba=balanced_accuracy_score(yqb, pq),
        qf1=f1_score(yqb, pq, average="macro", zero_division=0),
        auc=roc_auc_score(yqb[ok], prq[ok]) if len(np.unique(yqb[ok])) > 1 else float("nan"),
    )


def main() -> None:
    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    S = E.build_scores(data, real)
    sm = {c: st.source_of(c) for c in C.VIOLATIONS}
    print("Развёрнутая конфигурация:")
    print(f"  источники: {sm}")
    print(f"  режим порога: {C.THRESHOLD_MODE}")
    print(f"\nЧестная оценка, {N_SEEDS} разбиений (объединённые val-предсказания):\n")
    res = [run(data, real, S, sm, s) for s in range(N_SEEDS)]
    for key, title in (("macro", "Macro-F1 (4 метки организатора)"),
                       ("ba", "quality_class balanced accuracy"),
                       ("qf1", "quality_class macro-F1"),
                       ("auc", "quality_class ROC-AUC")):
        v = np.array([r[key] for r in res])
        print(f"  {title:36} {v.mean():.3f} ± {v.std():.3f}   ({v.min():.3f}..{v.max():.3f})")
    print("\n  по меткам организатора:")
    for lab in E.ORG_LABELS:
        v = np.array([r["per"][lab] for r in res])
        print(f"    {lab:34} {v.mean():.3f} ± {v.std():.3f}")


if __name__ == "__main__":
    main()
