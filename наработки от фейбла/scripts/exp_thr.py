# -*- coding: utf-8 -*-
"""Честное сравнение режимов порога (prior/f1/blend и ранговые kprior/kf1/kblend).

Парные StratifiedGroupKFold-разбиения по study_uid, порог/k подбирается ТОЛЬКО на
train-части, метрики по реальным снимкам. Разность режимов проверяется бутстрэпом
по фолдам (CI разности); принимаем только если CI не накрывает 0.

Запуск:
    python scripts/exp_thr.py --cnn out/cnn_resnet18,out/cnn_resnet34 --seeds 10
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dxa_qc"))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import exp_fusion as X  # noqa: E402
from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402

MODES = ("prior", "f1", "blend", "kprior", "kf1", "kblend")


def org_f1(data, fired, va):
    f1s = []
    for lab, crits in X.ORG_LABELS.items():
        gt = np.zeros(len(data), dtype=int)
        pr = np.zeros(len(data), dtype=int)
        for name in crits:
            gt |= np.nan_to_num(data["viol_" + name].values.astype(float), nan=0.0).astype(int)
            pr |= fired[:, C.VIOLATION_IDX[name]]
        f1s.append(f1_score(gt[va], pr[va], zero_division=0))
    return float(np.mean(f1s))


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

    # карта по AUC (глобально) — та же, что в final_report
    auc_map = {}
    for j, name in enumerate(C.VIOLATIONS):
        y = data["viol_" + name].values.astype(float)
        ok = ~np.isnan(y) & real
        best_src, best_auc = "cnn", -1.0
        for s in X.SOURCES:
            okk = ok & ~np.isnan(S[s][:, j])
            if len(np.unique(y[okk].astype(int))) < 2:
                continue
            a = roc_auc_score(y[okk].astype(int), S[s][okk, j])
            if a > best_auc:
                best_auc, best_src = a, s
        auc_map[name] = best_src
    print("карта:", auc_map)

    idx = np.where(real)[0]
    g = data["study_uid"].values[idx]
    strat = data["region"].values[idx]

    # пофолдовые значения macro-F1 для каждого режима (одинаковые разбиения)
    fold_vals = {m: [] for m in MODES}
    for seed in range(args.seeds):
        sp = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr_i, va_i in sp.split(idx, strat, groups=g):
            tr, va = idx[tr_i], idx[va_i]
            for m in MODES:
                fired = X.fired_from(data, real, S, auc_map, tr, va, m)
                fold_vals[m].append(org_f1(data, fired, va))

    print(f"\n{'режим':10} {'macro-F1':>10} {'±std':>8}")
    for m in MODES:
        v = np.array(fold_vals[m])
        print(f"{m:10} {v.mean():10.3f} {v.std():8.3f}")

    # парные сравнения с prior и с blend
    print("\nпарные разности (бутстрэп по фолдам, 2000):")
    rng = np.random.default_rng(0)
    for base in ("prior", "blend"):
        for m in MODES:
            if m == base:
                continue
            diff = np.array(fold_vals[m]) - np.array(fold_vals[base])
            boots = [np.mean(rng.choice(diff, len(diff), replace=True)) for _ in range(2000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            sig = "ЗНАЧИМО" if (lo > 0 or hi < 0) else "шум"
            print(f"  {m:8} - {base:6}  Δ={diff.mean():+.3f} [{lo:+.3f};{hi:+.3f}]  {sig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
