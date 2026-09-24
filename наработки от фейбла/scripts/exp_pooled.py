# -*- coding: utf-8 -*-
"""Сводная таблица в метрике организатора (объединённые val-предсказания).

Для каждой карты источников и режима порога: 5-фолдовая схема по study_uid,
порог только по train-части, val-предсказания ОБЪЕДИНЯЮТСЯ (каждый реальный
снимок предсказан один раз), macro-F1 считается по 4 меткам на всём наборе —
как сделает организатор на отложенной выборке. Усреднение по 10 разбиениям.

Парные разности — бутстрэп по фолдам (принимаем только если CI не накрывает 0).

Запуск:
    python scripts/exp_pooled.py --cnn out/cnn_resnet18,out/cnn_resnet34 --seeds 10
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


def pooled_seed(data, real, S, source_map, seed, thr_mode):
    idx = np.where(real)[0]
    g = data["study_uid"].values[idx]
    strat = data["region"].values[idx]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fired = np.zeros((len(data), len(C.VIOLATIONS)), dtype=int)
    for tr_i, va_i in splitter.split(idx, strat, groups=g):
        tr, va = idx[tr_i], idx[va_i]
        f = X.fired_from(data, real, S, source_map, tr, va, thr_mode)
        fired[va] = f[va]
    per = {}
    for lab, crits in X.ORG_LABELS.items():
        gt = np.zeros(len(data), dtype=int)
        pr = np.zeros(len(data), dtype=int)
        for name in crits:
            gt |= np.nan_to_num(data["viol_" + name].values.astype(float), nan=0.0).astype(int)
            pr |= fired[:, C.VIOLATION_IDX[name]]
        per[lab] = f1_score(gt[real], pr[real], zero_division=0)
    return float(np.mean(list(per.values()))), per


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn", default="out/cnn_resnet18,out/cnn_resnet34")
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()

    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    dirs = [d.strip() for d in args.cnn.split(",") if d.strip()]
    oofs = [np.load(os.path.join(d, "oof_violation.npy")) for d in dirs]
    cnn_oof = np.nanmean(np.stack(oofs), axis=0)
    print(f"[pooled] CNN={dirs}")
    S = X.build_scores(data, real, cnn_oof)

    crits = list(C.VIOLATIONS)
    maps = {
        "team (cnn+axis-fused)": {"spine_positioning": "cnn", "spine_axis": "fused",
                                  "spine_artifacts": "cnn", "femur_positioning": "cnn",
                                  "femur_roi": "cnn"},
        "fused везде": {c: "fused" for c in crits},
    }
    auc_map = {}
    for j, name in enumerate(crits):
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
    maps["карта по AUC"] = auc_map
    print("карта по AUC:", auc_map)

    results = {}
    print(f"\n{'карта':24} {'порог':6} {'macro-F1':>10} {'±std':>8}   по меткам")
    for mname, sm in maps.items():
        for tmode in ("prior", "f1", "blend"):
            vals, pers = [], {k: [] for k in X.ORG_LABELS}
            for seed in range(args.seeds):
                m, per = pooled_seed(data, real, S, sm, seed, tmode)
                vals.append(m)
                for k in per:
                    pers[k].append(per[k])
            results[(mname, tmode)] = np.array(vals)
            per_s = " ".join(f"{k}={np.mean(v):.3f}" for k, v in pers.items())
            print(f"  {mname:22} {tmode:6} {np.mean(vals):10.3f} {np.std(vals):8.3f}   {per_s}")

    print("\nпарные разности (бутстрэп по 10 разбиениям):")
    rng = np.random.default_rng(0)

    def cmp(a, b, label):
        diff = results[a] - results[b]
        boots = [np.mean(rng.choice(diff, len(diff), replace=True)) for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        sig = "ЗНАЧИМО" if (lo > 0 or hi < 0) else "шум"
        print(f"  {label:44} Δ={diff.mean():+.3f} [{lo:+.3f};{hi:+.3f}]  {sig}")

    cmp(("карта по AUC", "prior"), ("team (cnn+axis-fused)", "prior"),
        "карта AUC - team (prior)")
    cmp(("карта по AUC", "blend"), ("team (cnn+axis-fused)", "prior"),
        "карта AUC + blend - team (prior)")
    cmp(("карта по AUC", "blend"), ("карта по AUC", "prior"),
        "карта AUC: blend - prior")
    cmp(("team (cnn+axis-fused)", "f1"), ("team (cnn+axis-fused)", "prior"),
        "team: f1 - prior")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
