# -*- coding: utf-8 -*-
"""Идеи внешнего советника (Astra): ранговые ансамбли и per-label пороги.

Проверяются на честной схеме (StratifiedGroupKFold по study_uid, порог по
train-части, метрики по реальным снимкам), с парными разбиениями и CI разности.

Идеи:
  #2 — per-label порог: prior (доля) vs f1-оптимум vs бленд;
  #4 — ранговый ансамбль источников (усредняем РАНГИ, не вероятности).

Запуск:
    python scripts/exp_astra.py --cnn out/cnn_resnet18,out/cnn_resnet34
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dxa_qc"))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import exp_fusion as X  # noqa: E402
from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402


def rank01(v):
    """Ранг -> [0,1], NaN сохраняются."""
    out = np.full(len(v), np.nan)
    ok = ~np.isnan(v)
    if ok.sum() == 0:
        return out
    r = rankdata(v[ok], method="average")
    out[ok] = (r - 1) / max(len(r) - 1, 1)
    return out


def add_rank_sources(S, data, real):
    """Добавить ранговые ансамбли cnn+geo и cnn+geobig по каждому критерию."""
    n = len(data)
    for name in ("rank_small", "rank_big"):
        S[name] = np.full((n, len(C.VIOLATIONS)), np.nan)
    for j, crit in enumerate(C.VIOLATIONS):
        a = rank01(S["cnn"][:, j])
        for src, out in (("geo", "rank_small"), ("geobig", "rank_big")):
            b = rank01(S[src][:, j])
            S[out][:, j] = np.nanmean(np.vstack([a, b]), axis=0)
    return S


def threshold_f1(y, p):
    grid = np.linspace(0.05, 0.95, 91)
    best, bt = -1.0, 0.5
    for t in grid:
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best:
            best, bt = f, float(t)
    return bt


def threshold_blend(y, p):
    return 0.5 * (threshold_f1(y, p) + X.prior_threshold(y, p))


THR_MODES = {"prior": X.prior_threshold, "f1": threshold_f1, "blend": threshold_blend}


def fired_with_mode(data, S, source_map, tr, va, thr_mode):
    fired = np.zeros((len(data), len(C.VIOLATIONS)), dtype=int)
    for j, name in enumerate(C.VIOLATIONS):
        key = "spine" if name.startswith("spine") else "femur"
        rmask = X.region_mask(data, key)
        y = data["viol_" + name].values.astype(float)
        p = S[source_map[name]][:, j]
        trc = tr[rmask[tr] & ~np.isnan(y[tr]) & ~np.isnan(p[tr])]
        if len(trc) < 2 or len(np.unique(y[trc].astype(int))) < 2:
            continue
        thr = THR_MODES[thr_mode](y[trc].astype(int), p[trc])
        va_ok = va[~np.isnan(p[va])]
        fired[va_ok[p[va_ok] >= thr], j] = 1
    return fired


def org_macro(data, fired, va):
    f1s = []
    for lab, crits in X.ORG_LABELS.items():
        gt = np.zeros(len(data), dtype=int)
        pr = np.zeros(len(data), dtype=int)
        for name in crits:
            gt |= np.nan_to_num(data["viol_" + name].values.astype(float), nan=0.0).astype(int)
            pr |= fired[:, C.VIOLATION_IDX[name]]
        f1s.append(f1_score(gt[va], pr[va], zero_division=0))
    return float(np.mean(f1s)), f1s


def eval_map(data, real, S, source_map, thr_mode, seeds=range(10)):
    idx = np.where(real)[0]
    g = data["study_uid"].values[idx]
    strat = data["region"].values[idx]
    macros, per_lab = [], []
    for seed in seeds:
        sp = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        folds = []
        for tr_i, va_i in sp.split(idx, strat, groups=g):
            tr, va = idx[tr_i], idx[va_i]
            fired = fired_with_mode(data, S, source_map, tr, va, thr_mode)
            m, f1s = org_macro(data, fired, va)
            folds.append(m)
            per_lab.append(f1s)
        macros.append(np.mean(folds))
    return float(np.mean(macros)), float(np.std(macros)), np.mean(np.array(per_lab), axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn", default="out/cnn_resnet18,out/cnn_resnet34")
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()

    manifest = ds.load_manifest(C.MANIFEST_CSV)
    data = fl.load_features(manifest)
    real = np.asarray(ds.real_mask(data))
    dirs = [d.strip() for d in args.cnn.split(",") if d.strip()]
    oofs = [np.load(os.path.join(d, "oof_violation.npy")) for d in dirs]
    cnn_oof = np.nanmean(np.stack(oofs), axis=0)
    print(f"[astra] CNN={dirs}")

    S = X.build_scores(data, real, cnn_oof)
    S = add_rank_sources(S, data, real)

    # карта источников по AUC (как в final_report) + ранговые источники в пуле
    pool = list(X.SOURCES) + ["rank_small", "rank_big"]
    auc_map = {}
    for j, name in enumerate(C.VIOLATIONS):
        y = data["viol_" + name].values.astype(float)
        ok = ~np.isnan(y) & real
        best_src, best_auc = "cnn", -1.0
        for s in pool:
            okk = ok & ~np.isnan(S[s][:, j])
            if len(np.unique(y[okk].astype(int))) < 2:
                continue
            from sklearn.metrics import roc_auc_score
            a = roc_auc_score(y[okk].astype(int), S[s][okk, j])
            if a > best_auc:
                best_auc, best_src = a, s
        auc_map[name] = best_src
        print(f"  {name:20} -> {best_src:11} AUC={best_auc:.3f}")

    base_map = {c: "fused" for c in C.VIOLATIONS}
    print("\n=== Идея #2: режим порога (карта по AUC) ===")
    for mode in ("prior", "f1", "blend"):
        m, sd, per = eval_map(data, real, S, auc_map, mode, seeds=range(args.seeds))
        print(f"  thr={mode:6} macro-F1={m:.3f}±{sd:.3f}  " +
              " ".join(f"{k}={v:.3f}" for k, v in zip(X.ORG_LABELS, per)))

    print("\n=== Идея #4: ранговый ансамбль vs вероятности (thr=prior) ===")
    for title, sm in (("fused везде", base_map), ("карта по AUC", auc_map)):
        m, sd, per = eval_map(data, real, S, sm, "prior", seeds=range(args.seeds))
        print(f"  {title:14} macro-F1={m:.3f}±{sd:.3f}  " +
              " ".join(f"{k}={v:.3f}" for k, v in zip(X.ORG_LABELS, per)))

    print("\n=== Парные сравнения (10 сидов, разность по фолдам) ===")
    def paired(sm_a, mode_a, sm_b, mode_b, label):
        idx = np.where(real)[0]
        g = data["study_uid"].values[idx]
        strat = data["region"].values[idx]
        da, db = [], []
        for seed in range(args.seeds):
            sp = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            for tr_i, va_i in sp.split(idx, strat, groups=g):
                tr, va = idx[tr_i], idx[va_i]
                fa = fired_with_mode(data, S, sm_a, tr, va, mode_a)
                fb = fired_with_mode(data, S, sm_b, tr, va, mode_b)
                da.append(org_macro(data, fa, va)[0])
                db.append(org_macro(data, fb, va)[0])
        da, db = np.array(da), np.array(db)
        diff = da - db
        rng = np.random.default_rng(0)
        boots = [np.mean(rng.choice(diff, len(diff), replace=True)) for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"  {label:38} Δ={diff.mean():+.3f} [{lo:+.3f};{hi:+.3f}]  "
              f"{'ЗНАЧИМО' if lo > 0 or hi < 0 else 'шум'}")

    paired(auc_map, "prior", base_map, "prior", "карта AUC - fused везде")
    paired(auc_map, "f1", auc_map, "prior", "карта AUC: thr f1 - prior")
    paired(auc_map, "blend", auc_map, "prior", "карта AUC: thr blend - prior")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
