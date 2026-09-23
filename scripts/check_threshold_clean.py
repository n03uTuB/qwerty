# -*- coding: utf-8 -*-
"""Чистое сравнение режимов порога (без переиспользования OOF-скоров).

Проверка предыдущего вывода «f1 лучше prior». Возможная слабость: в честной CV
скор train-части — это OOF из того же пула, что и val, и режим f1 (подгонка под
распределение) мог извлечь из этого выгоду. Здесь источник — только геометрия
(логистическая регрессия), поэтому скоры для train-части считаются ВНУТРИ фолда
(вложенная CV), полностью без утечки: ни один val-снимок не влияет на скоры train.

Сравниваются режимы prior (k = доля * n) и f1 (оптимум F1 на train).

Запуск:
    cd dxa_qc && python ../scripts/check_threshold_clean.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dxa_qc"))

from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402
from src import stack as st        # noqa: E402
from src.train import best_threshold, prior_threshold  # noqa: E402

import eval_honest_organizer as E  # noqa: E402

N_SEEDS = 10


def _clf():
    return Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(max_iter=2000,
                                              class_weight="balanced", C=1.0))])


def _inner_oof(X, y, groups, n_inner=5):
    oof = np.full(len(y), np.nan)
    for tr, va in GroupKFold(n_splits=n_inner).split(X, y, groups=groups):
        if len(np.unique(y[tr])) < 2:
            continue
        m = _clf()
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    return oof


def run(data, real, seed, mode):
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
            cols = [c for c in st.CRITERION_FEATURES.get(name, []) if c in data.columns]
            trc = tr[rmask[tr] & ~np.isnan(y[tr])]
            vac = va[rmask[va] & ~np.isnan(y[va])]
            if len(trc) < 10 or len(vac) == 0 or len(np.unique(y[trc].astype(int))) < 2:
                continue
            Xtr = data.loc[trc, cols].fillna(0).values if cols else np.zeros((len(trc), 1))
            Xva = data.loc[vac, cols].fillna(0).values if cols else np.zeros((len(vac), 1))
            yt = y[trc].astype(int)
            # скоры для train — вложенная CV внутри фолда (чисто)
            p_tr = _inner_oof(Xtr, yt, data["study_uid"].values[trc])
            ok = ~np.isnan(p_tr)
            if ok.sum() < 5 or len(np.unique(yt[ok])) < 2:
                continue
            thr = (prior_threshold(yt[ok], p_tr[ok]) if mode == "prior"
                   else best_threshold(yt[ok], p_tr[ok])[0])
            m = _clf()
            m.fit(Xtr, yt)
            p_va = m.predict_proba(Xva)[:, 1]
            fired[vac[p_va >= thr], j] = 1
    fs = []
    for crits in E.ORG_LABELS.values():
        gt = np.zeros(len(data), dtype=int)
        pr = np.zeros(len(data), dtype=int)
        for name in crits:
            v = data["viol_" + name].values.astype(float)
            gt |= np.nan_to_num(v, nan=0.0).astype(int)
            pr |= fired[:, C.VIOLATION_IDX[name]]
        fs.append(E.f1_score(gt[idx], pr[idx], zero_division=0))
    return float(np.mean(fs))


def main() -> None:
    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    print("Чистое (вложенное) сравнение режимов порога, источник — геометрия.")
    print(f"{N_SEEDS} разбиений:\n")
    print(f"{'режим':10} {'средн':>7} {'медиана':>8} {'ст.откл':>8}  мин..макс")
    for mode in ("prior", "f1"):
        vals = np.array([run(data, real, s, mode) for s in range(N_SEEDS)])
        print(f"{mode:10} {vals.mean():7.3f} {np.median(vals):8.3f} "
              f"{vals.std():8.3f}  {vals.min():.2f}..{vals.max():.2f}")


if __name__ == "__main__":
    main()
