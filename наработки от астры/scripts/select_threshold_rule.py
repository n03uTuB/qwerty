# -*- coding: utf-8 -*-
"""Выбор правила порога: prior / f1 / бленды, с разбивкой по меткам.

f1-порог (оптимум F1 на train-части) в честной CV устойчиво лучше prior. Здесь
проверяем ещё компромиссные правила и смотрим, за счёт каких меток идёт выигрыш:
  prior    — k = доля * n;
  alpha1.5 — k = 1.5 * доля * n;
  f1       — оптимум F1 на train;
  f1_hi    — то же, но при равенстве F1 берётся ВЕРХНИЙ порог (осторожнее);
  blend    — среднее prior и f1.

Запуск:
    cd dxa_qc && python ../scripts/select_threshold_rule.py
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
from src.train import prior_threshold  # noqa: E402
from sklearn.metrics import f1_score as _f1  # noqa: E402

import eval_honest_organizer as E  # noqa: E402

N_SEEDS = 20
THS = np.linspace(0.02, 0.98, 97)


def _f1_opt(y, p, high=False):
    best, bt = -1.0, 0.5
    order = THS[::-1] if high else THS
    for t in order:
        f = _f1(y, (p >= t).astype(int), zero_division=0)
        if f > best:
            best, bt = f, float(t)
    return bt


def _alpha(y, p, a):
    prev = float(np.mean(y)) if len(y) else 0.0
    if prev <= 0 or prev >= 1 or len(p) == 0:
        return 0.5
    k = min(max(1, int(round(a * prev * len(p)))), len(p))
    return float(np.sort(p)[::-1][k - 1])


def _rule(mode, y, p):
    if mode == "prior":
        return prior_threshold(y, p)
    if mode == "alpha1.5":
        return _alpha(y, p, 1.5)
    if mode == "f1":
        return _f1_opt(y, p)
    if mode == "f1_hi":
        return _f1_opt(y, p, high=True)
    if mode == "blend":
        return 0.5 * (prior_threshold(y, p) + _f1_opt(y, p))
    raise ValueError(mode)


def run(data, real, S, sm, seed, mode):
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
            thr = _rule(mode, y[trc].astype(int), p[trc])
            va_ok = va[~np.isnan(p[va])]
            fired[va_ok[p[va_ok] >= thr], j] = 1
    per_label, fs = {}, []
    for lab, crits in E.ORG_LABELS.items():
        gt = np.zeros(len(data), dtype=int)
        pr = np.zeros(len(data), dtype=int)
        for name in crits:
            v = data["viol_" + name].values.astype(float)
            gt |= np.nan_to_num(v, nan=0.0).astype(int)
            pr |= fired[:, C.VIOLATION_IDX[name]]
        f = _f1(gt[idx], pr[idx], zero_division=0)
        per_label[lab] = f
        fs.append(f)
    return float(np.mean(fs)), per_label


def main() -> None:
    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    S = E.build_scores(data, real)
    sm = {c: st.source_of(c) for c in C.VIOLATIONS}
    modes = ["prior", "alpha1.5", "f1", "f1_hi", "blend"]
    print(f"Конфигурация: {sm}")
    print(f"Честная macro-F1 (4 метки), {N_SEEDS} разбиений, с разбивкой по меткам:\n")
    print(f"{'правило':10} {'macro-F1':>9} {'ст.откл':>8}   "
          + "  ".join(f"{k:>8}" for k in E.ORG_LABELS))
    for mode in modes:
        res = [run(data, real, S, sm, s, mode) for s in range(N_SEEDS)]
        macros = np.array([r[0] for r in res])
        per = {k: np.mean([r[1][k] for r in res]) for k in E.ORG_LABELS}
        print(f"{mode:10} {macros.mean():9.3f} {macros.std():8.3f}   "
              + "  ".join(f"{per[k]:8.3f}" for k in E.ORG_LABELS))


if __name__ == "__main__":
    main()
