# -*- coding: utf-8 -*-
"""Сводное честное сравнение: карта источников × режим порога.

Один харнесс (StratifiedGroupKFold по study_uid, порог только по train-части,
метрики по реальным снимкам, синтетика только в обучении). Показывает вклад
каждого решения (источник по критерию, режим порога) и парно сравнивает с базой
«текущая конфигурация + prior».

Запуск:
    python scripts/exp_final.py --cnn out/cnn_resnet18,out/cnn_resnet34 --seeds 10
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dxa_qc"))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import exp_fusion as X  # noqa: E402
from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402


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
    print(f"[final] CNN-ансамбль: {dirs}")
    S = X.build_scores(data, real, cnn_oof)

    crits = list(C.VIOLATIONS)
    maps = {
        "team (cnn+axis-fused)": {"spine_positioning": "cnn", "spine_axis": "fused",
                                  "spine_artifacts": "cnn", "femur_positioning": "cnn",
                                  "femur_roi": "cnn"},
        "fused везде": {c: "fused" for c in crits},
        "fusedbig везде": {c: "fusedbig" for c in crits},
    }
    # карта по AUC
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

    thr_modes = ("prior", "f1", "blend")
    results = {}
    for mname, sm in maps.items():
        for tmode in thr_modes:
            macro, std, per, q = X.eval_fixed_seeds(
                data, real, S, sm, seeds=range(args.seeds), thr_mode=tmode)
            results[(mname, tmode)] = (macro, std, per, q)
            per_s = " ".join(f"{k}={v:.3f}" for k, v in per.items())
            print(f"  {mname:24} {tmode:6}  macro={macro:.3f}±{std:.3f}  {per_s}")

    # --- парные сравнения с базой (team, prior) ---
    print("\nпарные разности против базы 'team + prior' (10 разбиений, бутстрэп):")
    from sklearn.model_selection import StratifiedGroupKFold
    idx = np.where(real)[0]
    g = data["study_uid"].values[idx]
    strat = data["region"].values[idx]

    def fold_macros(sm, tmode):
        vals = []
        for seed in range(args.seeds):
            sp = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            for tr_i, va_i in sp.split(idx, strat, groups=g):
                tr, va = idx[tr_i], idx[va_i]
                fired = X.fired_from(data, real, S, sm, tr, va, tmode)
                vals.append(X.org_scores(data, fired, va))
        return vals

    def macro_of(list_of_dicts):
        return np.array([np.mean(list(d.values())) for d in list_of_dicts])

    base = macro_of(fold_macros(maps["team (cnn+axis-fused)"], "prior"))
    rng = np.random.default_rng(0)
    for mname in ("карта по AUC",):
        for tmode in thr_modes:
            cur = macro_of(fold_macros(maps[mname], tmode))
            diff = cur - base
            boots = [np.mean(rng.choice(diff, len(diff), replace=True)) for _ in range(2000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            sig = "ЗНАЧИМО" if (lo > 0 or hi < 0) else "шум"
            print(f"  {mname} + {tmode:6} - base: Δ={diff.mean():+.3f} "
                  f"[{lo:+.3f};{hi:+.3f}]  {sig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
