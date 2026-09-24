# -*- coding: utf-8 -*-
"""Диагностика по меткам организатора: где теряется macro-F1.

Для каждой из 4 меток показывает (усреднённо по сидам, честно) precision/recall/F1
при разных правилах порога и «идеальном» top-k. Помогает понять, что улучшать:
ранжирование (AUC) или правило порога.

Запуск:
    python scripts/diag_labels.py --cnn out/cnn_resnet18,out/cnn_resnet34
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
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


def prf(gt, pr):
    tp = int(((gt == 1) & (pr == 1)).sum())
    fp = int(((gt == 0) & (pr == 1)).sum())
    fn = int(((gt == 1) & (pr == 0)).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f = 2 * p * r / max(p + r, 1e-9)
    return p, r, f


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
    S = X.build_scores(data, real, cnn_oof)

    # карта по AUC (глобально)
    auc_map = {}
    for j, name in enumerate(C.VIOLATIONS):
        y = data["viol_" + name].values.astype(float)
        ok = ~np.isnan(y) & real
        best_src, best_auc = "cnn", -1.0
        for s in X.SOURCES:
            okk = ok & ~np.isnan(S[s][:, j])
            if len(np.unique(y[okk].astype(int))) < 2:
                continue
            from sklearn.metrics import roc_auc_score
            a = roc_auc_score(y[okk].astype(int), S[s][okk, j])
            if a > best_auc:
                best_auc, best_src = a, s
        auc_map[name] = best_src
    print("карта:", auc_map)

    idx = np.where(real)[0]
    g = data["study_uid"].values[idx]
    strat = data["region"].values[idx]

    modes = ("prior", "f1", "oracle_f1")
    agg = {lab: {m: [] for m in modes} for lab in X.ORG_LABELS}
    for seed in range(args.seeds):
        sp = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr_i, va_i in sp.split(idx, strat, groups=g):
            tr, va = idx[tr_i], idx[va_i]
            # скоры на валидации по каждому критерию
            fired = {m: np.zeros((len(data), len(C.VIOLATIONS)), dtype=int) for m in modes}
            for j, name in enumerate(C.VIOLATIONS):
                key = "spine" if name.startswith("spine") else "femur"
                rmask = X.region_mask(data, key)
                y = data["viol_" + name].values.astype(float)
                p = S[auc_map[name]][:, j]
                trc = tr[rmask[tr] & ~np.isnan(y[tr]) & ~np.isnan(p[tr])]
                if len(trc) < 2 or len(np.unique(y[trc].astype(int))) < 2:
                    continue
                ytr, ptr = y[trc].astype(int), p[trc]
                va_ok = va[~np.isnan(p[va])]
                pv = p[va_ok]
                thr_prior = X.prior_threshold(ytr, ptr)
                grid = np.linspace(0.05, 0.95, 91)
                thr_f1 = float(grid[int(np.argmax([f1_score(ytr, (ptr >= t).astype(int),
                                                           zero_division=0) for t in grid]))])
                fired["prior"][va_ok[pv >= thr_prior], j] = 1
                fired["f1"][va_ok[pv >= thr_f1], j] = 1
                # oracle: порог, лучший по F1 НА ВАЛИДАЦИИ (нечестно, верхняя граница)
                yv = y[va_ok].astype(int)
                if len(np.unique(yv)) > 1:
                    thr_o = float(grid[int(np.argmax([f1_score(yv, (pv >= t).astype(int),
                                                               zero_division=0) for t in grid]))])
                    fired["oracle_f1"][va_ok[pv >= thr_o], j] = 1
            for lab, crits in X.ORG_LABELS.items():
                gt = np.zeros(len(data), dtype=int)
                for name in crits:
                    gt |= np.nan_to_num(data["viol_" + name].values.astype(float),
                                        nan=0.0).astype(int)
                for m in modes:
                    pr = np.zeros(len(data), dtype=int)
                    for name in crits:
                        pr |= fired[m][:, C.VIOLATION_IDX[name]]
                    agg[lab][m].append(prf(gt[va], pr[va]))

    print(f"\n{'метка':14} {'режим':10} {'P':>6} {'R':>6} {'F1':>6}")
    for lab in X.ORG_LABELS:
        for m in modes:
            arr = np.array(agg[lab][m])
            print(f"{lab:14} {m:10} {arr[:,0].mean():6.3f} {arr[:,1].mean():6.3f} "
                  f"{arr[:,2].mean():6.3f}")
        # macro вклад
        print()
    print("macro-F1 по режимам:")
    for m in modes:
        macro = np.mean([np.mean(np.array(agg[lab][m])[:, 2]) for lab in X.ORG_LABELS])
        print(f"  {m:10} {macro:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
