# -*- coding: utf-8 -*-
"""Честная оценка с «лучшими» геомоделями зоопарка (расширенный кэш признаков).

Зоопарк (scripts/exp_models.py) нашёл сильные пары (набор признаков, модель):
  spine_positioning  V3(spine_iliac_signal, spine_bottom_cut) + RF
  spine_axis         V3(spine_midline_deg) + RF
  spine_artifacts    V3res(ribs, peaks, midline_residual) + LR C=0.1
  femur_positioning  margins(margin_min_cm, width_cm) + ET
  femur_roi          V3(height_cm, bone_ratio, margin_min_cm) + LR C=0.1

Здесь они считаются честно (GroupKFold по study_uid), объединяются с CNN-ансамблем
и оцениваются в метрике организатора (объединённые val-предсказания, 10 разбиений),
с парными CI разности против базовой конфигурации.

Запуск:
    python scripts/exp_geo_best.py --cnn out/cnn_resnet18,out/cnn_resnet34 --seeds 10
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dxa_qc"))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import exp_fusion as X  # noqa: E402
from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402

CACHE = os.path.join("out", "feat_cache.csv")

# лучшая пара (набор, модель) из зоопарка
ZOO_BEST = {
    "spine_positioning": (["spine_iliac_signal", "spine_bottom_cut"], "RF"),
    "spine_axis": (["spine_midline_deg"], "RF"),
    "spine_artifacts": (["spine_ribs_signal", "spine_vertebra_peaks",
                         "spine_midline_residual"], "LR01"),
    "femur_positioning": (["femur_margin_min_cm", "femur_width_cm"], "ET"),
    "femur_roi": (["femur_height_cm", "femur_bone_ratio", "femur_margin_min_cm"], "LR01"),
}


def factory(name):
    if name == "RF":
        return lambda: RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                              random_state=0, n_jobs=-1)
    if name == "ET":
        return lambda: ExtraTreesClassifier(n_estimators=400, class_weight="balanced_subsample",
                                            random_state=0, n_jobs=-1)
    if name == "LR01":
        return lambda: Pipeline([("s", StandardScaler()),
                                 ("c", LogisticRegression(max_iter=2000,
                                                          class_weight="balanced", C=0.1))])
    raise ValueError(name)


def cv_score(X, y, groups, fac, n_splits=5):
    oof = np.full(len(y), np.nan)
    for tr, va in GroupKFold(n_splits=n_splits).split(X, groups=groups):
        if len(np.unique(y[tr])) < 2:
            continue
        m = fac().fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    return oof


def build_scores(data, real, cnn_oof):
    n = len(data)
    S = {s: np.full((n, len(C.VIOLATIONS)), np.nan) for s in ("cnn", "geoZ", "fusedZ")}
    for j, name in enumerate(C.VIOLATIONS):
        key = "spine" if name.startswith("spine") else "femur"
        rmask = X.region_mask(data, key)
        y = data["viol_" + name].values.astype(float)
        known = ~np.isnan(y) & rmask & real
        if known.sum() == 0:
            continue
        S["cnn"][known, j] = cnn_oof[known, j]
        cols, model = ZOO_BEST[name]
        cols = [c for c in cols if c in data.columns]
        Xg = data.loc[known, cols].fillna(0).values
        g = data.loc[known, "study_uid"].values
        S["geoZ"][known, j] = cv_score(Xg, y[known].astype(int), g, factory(model))
        Xf = np.column_stack([cnn_oof[known, j], Xg])
        S["fusedZ"][known, j] = cv_score(Xf, y[known].astype(int), g, factory(model))
    return S


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

    man = ds.load_manifest(C.MANIFEST_CSV)
    rich = pd.read_csv(CACHE)
    data = man.merge(rich, on="image_uid", how="left", suffixes=("", "_r"))
    if "study_uid_r" in data.columns:
        data["study_uid"] = data["study_uid"].fillna(data["study_uid_r"])
    real = np.asarray(ds.real_mask(data))

    dirs = [d.strip() for d in args.cnn.split(",") if d.strip()]
    oofs = [np.load(os.path.join(d, "oof_violation.npy")) for d in dirs]
    cnn_oof = np.nanmean(np.stack(oofs), axis=0)
    print(f"[geo_best] строк={len(data)} реальных={int(real.sum())} CNN={dirs}")

    S = build_scores(data, real, cnn_oof)
    crits = list(C.VIOLATIONS)

    # AUC источников
    for j, name in enumerate(crits):
        y = data["viol_" + name].values.astype(float)
        ok = ~np.isnan(y) & real
        line = f"  {name:20} "
        for s in ("cnn", "geoZ", "fusedZ"):
            okk = ok & ~np.isnan(S[s][:, j])
            line += f"{s}={roc_auc_score(y[okk].astype(int), S[s][okk, j]):.3f}  "
        print(line)

    maps = {
        "база (cnn + axis fusedZ)": {c: ("fusedZ" if c == "spine_axis" else "cnn") for c in crits},
    }
    auc_map = {}
    for j, name in enumerate(crits):
        y = data["viol_" + name].values.astype(float)
        ok = ~np.isnan(y) & real
        best_src, best_auc = "cnn", -1.0
        for s in ("cnn", "geoZ", "fusedZ"):
            okk = ok & ~np.isnan(S[s][:, j])
            if len(np.unique(y[okk].astype(int))) < 2:
                continue
            a = roc_auc_score(y[okk].astype(int), S[s][okk, j])
            if a > best_auc:
                best_auc, best_src = a, s
        auc_map[name] = best_src
    maps["карта по AUC (cnn/geoZ/fusedZ)"] = auc_map
    print("карта по AUC:", auc_map)

    results = {}
    print(f"\n{'карта':34} {'порог':6} {'macro-F1':>10} {'±std':>8}   по меткам")
    for mname, sm in maps.items():
        for tmode in ("prior", "blend"):
            vals, pers = [], {k: [] for k in X.ORG_LABELS}
            for seed in range(args.seeds):
                m, per = pooled_seed(data, real, S, sm, seed, tmode)
                vals.append(m)
                for k in per:
                    pers[k].append(per[k])
            results[(mname, tmode)] = np.array(vals)
            per_s = " ".join(f"{k}={np.mean(v):.3f}" for k, v in pers.items())
            print(f"  {mname:32} {tmode:6} {np.mean(vals):10.3f} {np.std(vals):8.3f}   {per_s}")

    print("\nпарные разности (бутстрэп по 10 разбиениям):")
    rng = np.random.default_rng(0)

    def cmp(a, b, label):
        diff = results[a] - results[b]
        boots = [np.mean(rng.choice(diff, len(diff), replace=True)) for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        sig = "ЗНАЧИМО" if (lo > 0 or hi < 0) else "шум"
        print(f"  {label:52} Δ={diff.mean():+.3f} [{lo:+.3f};{hi:+.3f}]  {sig}")

    base = ("база (cnn + axis fusedZ)", "prior")
    cmp(("карта по AUC (cnn/geoZ/fusedZ)", "prior"), base, "карта AUC - база (prior)")
    cmp(("карта по AUC (cnn/geoZ/fusedZ)", "blend"), base, "карта AUC + blend - база (prior)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
