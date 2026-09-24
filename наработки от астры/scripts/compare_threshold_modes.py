# -*- coding: utf-8 -*-
"""Честное сравнение режимов порога: prior / f1 / nested.

Режим ``prior`` (порог по доле нарушений из train) объявлен дефолтом как самый
устойчивый. Проверяем это ЧЕСТНО на текущей конфигурации источников: порог
подбирается только по train-части фолда, метрика — по val, 20 разбиений.

  prior  — k = round(доля * n), порог = k-й по величине скор;
  f1     — порог, максимизирующий F1 на train (оптимистичен внутри фолда);
  nested — порог по внутренней CV на train (самый консервативный).

Запуск:
    cd dxa_qc && python ../scripts/compare_threshold_modes.py
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
from src.train import best_threshold, prior_threshold  # noqa: E402

import eval_honest_organizer as E  # noqa: E402

N_SEEDS = 20


def _nested_thr(y, p, groups, n_inner=4):
    """Порог по внутренней CV: максимизирует F1 на внутренних фолдах."""
    if len(np.unique(y)) < 2:
        return 0.5
    ths = np.linspace(0.05, 0.95, 19)
    agg = np.zeros_like(ths)
    cnt = 0
    for tr, va in GroupKFold(n_splits=n_inner).split(p.reshape(-1, 1), y, groups=groups):
        if len(np.unique(y[tr])) < 2:
            continue
        thr_i, _ = best_threshold(y[tr], p[tr])
        agg += np.array([1.0 if thr_i >= t else 0.0 for t in ths])
        cnt += 1
    if cnt == 0:
        return 0.5
    return float(ths[int(np.argmax(agg))])


def _threshold(mode, y, p, groups):
    if mode == "prior":
        return prior_threshold(y, p)
    if mode == "f1":
        return best_threshold(y, p)[0]
    return _nested_thr(y, p, groups)


def macro_f1(data, real, S, source_map, seed, mode):
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
            thr = _threshold(mode, y[trc].astype(int), p[trc],
                             data["study_uid"].values[trc])
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
    print(f"Конфигурация источников: {sm}\n")
    print(f"{'режим порога':14} {'средн':>7} {'медиана':>8} {'ст.откл':>8}  мин..макс")
    for mode in ("prior", "f1", "nested"):
        vals = np.array([macro_f1(data, real, S, sm, s, mode) for s in range(N_SEEDS)])
        print(f"{mode:14} {vals.mean():7.3f} {np.median(vals):8.3f} "
              f"{vals.std():8.3f}  {vals.min():.2f}..{vals.max():.2f}")


if __name__ == "__main__":
    main()
