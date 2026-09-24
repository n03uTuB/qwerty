# -*- coding: utf-8 -*-
"""Дешёвые идеи Astra: агрегация по study, робастный порог, rank-усреднение.

Сравнение на честной схеме (порог по train-части фолда, объединённые val-предсказания,
20 разбиений StratifiedGroupKFold по study_uid). Базовая конфигурация —
config.CRITERION_SOURCES + config.THRESHOLD_MODE.

Варианты:
  base     — как в пайплайне;
  study    — скор снимка заменяется средним скором его исследования (шумоподавление);
  rthr     — робастный порог: медиана f1-оптимумов по бутстрэп-подвыборкам train;
  rankavg  — для критериев, где есть и CNN, и геометрия, скоры усредняются по рангам.

Запуск:
    cd dxa_qc && python ../scripts/exp_cheap_ideas.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dxa_qc"))

from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402
from src import stack as st        # noqa: E402
from src.train import best_threshold, pick_threshold, prior_threshold  # noqa: E402

import eval_honest_organizer as E  # noqa: E402

N_SEEDS = 20
RNG = np.random.default_rng(0)


def _study_mean(S, data, real, j, key):
    """Заменить скор снимка средним скором его исследования (внутри области)."""
    out = S.copy()
    rm = data["region"].str.contains(key).values
    idx = np.where(rm & real & ~np.isnan(out[:, j]))[0]
    for s in np.unique(data["study_uid"].values[idx]):
        sel = idx[data["study_uid"].values[idx] == s]
        out[sel, j] = float(np.nanmean(out[sel, j]))
    return out


def _rank_avg(a, b):
    """Среднее рангов (нормировано в [0,1]) — калибровки источников несопоставимы."""
    ok = ~np.isnan(a) & ~np.isnan(b)
    out = np.full(len(a), np.nan)
    if ok.sum() < 2:
        return out
    ra = rankdata(a[ok]) / ok.sum()
    rb = rankdata(b[ok]) / ok.sum()
    out[ok] = 0.5 * (ra + rb)
    return out


def _robust_thr(y, p, n_boot=60):
    """Медиана f1-оптимумов по бутстрэп-подвыборкам train (устойчивее argmax)."""
    if len(np.unique(y)) < 2:
        return 0.5
    ths = []
    for _ in range(n_boot):
        idx = RNG.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        ths.append(best_threshold(y[idx], p[idx])[0])
    return float(np.median(ths)) if ths else best_threshold(y, p)[0]


def run(data, real, S, sm, seed, variant):
    idx = np.where(real)[0]
    g = data["study_uid"].values[idx]
    strat = data["region_idx"].values[idx] if "region_idx" in data.columns else None
    if strat is None or len(np.unique(strat)) < 2:
        splits = GroupKFold(n_splits=5).split(idx, groups=g)
    else:
        splits = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                      random_state=seed).split(idx, strat, groups=g)
    n = len(data)
    # базовая матрица скоров [n, 5] по конфигурации источников
    M = np.full((n, len(C.VIOLATIONS)), np.nan)
    for j, name in enumerate(C.VIOLATIONS):
        M[:, j] = S[sm[name]][:, j]

    if variant == "study":
        for j, name in enumerate(C.VIOLATIONS):
            key = "spine" if name.startswith("spine") else "femur"
            M = _study_mean(M, data, real, j, key)
    elif variant == "rankavg":
        for j in range(len(C.VIOLATIONS)):
            M[:, j] = _rank_avg(S["cnn"][:, j], S["geo"][:, j])

    fired = np.zeros((len(data), len(C.VIOLATIONS)), dtype=int)
    for tr_i, va_i in splits:
        tr, va = idx[tr_i], idx[va_i]
        for j, name in enumerate(C.VIOLATIONS):
            key = "spine" if name.startswith("spine") else "femur"
            rmask = data["region"].str.contains(key).values
            y = data["viol_" + name].values.astype(float)
            p = M[:, j]
            trc = tr[rmask[tr] & ~np.isnan(y[tr]) & ~np.isnan(p[tr])]
            if len(trc) < 2 or len(np.unique(y[trc].astype(int))) < 2:
                continue
            yt, pt = y[trc].astype(int), p[trc]
            if variant == "rthr":
                thr = _robust_thr(yt, pt)
            else:
                thr, _ = pick_threshold(yt, pt, C.THRESHOLD_MODE)
            va_ok = va[~np.isnan(p[va])]
            fired[va_ok[p[va_ok] >= thr], j] = 1
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
    return float(np.mean(fs)), per


def main() -> None:
    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    S = E.build_scores(data, real)
    sm = {c: st.source_of(c) for c in C.VIOLATIONS}
    print(f"База: источники {sm}, режим {C.THRESHOLD_MODE}\n")
    print(f"{'вариант':10} {'macro-F1':>10} {'ст.откл':>8}   "
          + "  ".join(f"{k:>8}" for k in E.ORG_LABELS))
    for variant in ("base", "study", "rthr", "rankavg"):
        res = [run(data, real, S, sm, s, variant) for s in range(N_SEEDS)]
        m = np.array([r[0] for r in res])
        per = {k: np.mean([r[1][k] for r in res]) for k in E.ORG_LABELS}
        print(f"{variant:10} {m.mean():10.3f} {m.std():8.3f}   "
              + "  ".join(f"{per[k]:8.3f}" for k in E.ORG_LABELS))


if __name__ == "__main__":
    main()
