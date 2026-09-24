# -*- coding: utf-8 -*-
"""Класс модели на критерий: линейная логистика vs нелинейные (RF/GBM/SVM/полином).

Мотив: у ротации бедра зависимость U-образная (недоротация и переротация — оба
брак), а линейная логистика её не описывает. Проверяем, поднимает ли нелинейная
модель AUC критерия и итоговую macro-F1.

Запуск:
    python dxa_qc_work/scripts/exp_model_class.py --seeds 10
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import _common  # noqa: F401

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402
from src.train import pick_threshold  # noqa: E402

from exp_sweep import ORG_LABELS  # noqa: E402
from build_v3 import V3_FEATURES, V3_SOURCES  # noqa: E402


def make_clf(kind: str):
    if kind == "lr":
        return Pipeline([("s", StandardScaler()),
                         ("c", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    if kind == "poly":
        return Pipeline([("p", PolynomialFeatures(2, include_bias=False)),
                         ("s", StandardScaler()),
                         ("c", LogisticRegression(max_iter=3000, class_weight="balanced"))])
    if kind == "rf":
        return RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                      class_weight="balanced", random_state=0, n_jobs=1)
    if kind == "gbm":
        return GradientBoostingClassifier(n_estimators=200, max_depth=2,
                                          learning_rate=0.05, random_state=0)
    if kind == "svm":
        return Pipeline([("s", StandardScaler()),
                         ("c", SVC(C=2.0, gamma="scale", class_weight="balanced",
                                   probability=True, random_state=0))])
    raise ValueError(kind)


def cv_score(data, real, name, cols, source, kind, oof_cnn):
    """OOF-скоры критерия выбранным классификатором."""
    j = C.VIOLATION_IDX[name]
    reg = "spine" if name.startswith("spine") else "femur"
    rmask = data["region"].str.contains(reg).values & real
    y = data["viol_" + name].values.astype(float)
    known = ~np.isnan(y) & rmask
    out = np.full(len(data), np.nan)
    cnn = oof_cnn[known, j]
    if source == "cnn":
        out[known] = cnn
        return out
    X = data.loc[known, cols].fillna(0).values
    Xf = np.column_stack([cnn, X]) if source == "fused" else X
    yk = y[known].astype(int)
    gk = data.loc[known, "study_uid"].values
    if len(np.unique(yk)) < 2:
        return out
    for tr, va in GroupKFold(n_splits=5).split(Xf, yk, groups=gk):
        if len(np.unique(yk[tr])) < 2:
            continue
        m = make_clf(kind).fit(Xf[tr], yk[tr])
        out[known][va] = m.predict_proba(Xf[va])[:, 1]
    return out


def honest(data, real, S, n_seeds):
    idx = np.where(real)[0]
    g = data["study_uid"].values[idx]
    strat = data["region_idx"].values[idx]
    macros = []
    for seed in range(n_seeds):
        splits = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                      random_state=seed).split(idx, strat, groups=g)
        fired = np.zeros((len(data), len(C.VIOLATIONS)), dtype=int)
        for tr_i, va_i in splits:
            tr, va = idx[tr_i], idx[va_i]
            for j, name in enumerate(C.VIOLATIONS):
                reg = "spine" if name.startswith("spine") else "femur"
                rmask = data["region"].str.contains(reg).values
                y = data["viol_" + name].values.astype(float)
                p = S[name]
                trc = tr[rmask[tr] & ~np.isnan(y[tr]) & ~np.isnan(p[tr])]
                if len(trc) < 2 or len(np.unique(y[trc].astype(int))) < 2:
                    continue
                thr, _ = pick_threshold(y[trc].astype(int), p[trc], C.THRESHOLD_MODE)
                va_ok = va[~np.isnan(p[va])]
                fired[va_ok[p[va_ok] >= thr], j] = 1
        fs = []
        for lab, crits in ORG_LABELS.items():
            gt = np.zeros(len(data), dtype=int)
            pr = np.zeros(len(data), dtype=int)
            for name in crits:
                gt |= np.nan_to_num(data["viol_" + name].values.astype(float), nan=0.0).astype(int)
                pr |= fired[:, C.VIOLATION_IDX[name]]
            fs.append(f1_score(gt[idx], pr[idx], zero_division=0))
        macros.append(float(np.mean(fs)))
    return float(np.mean(macros)), float(np.std(macros))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(_common.OUT, "exp_model_class.json"))
    args = ap.parse_args()

    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    oof_cnn = np.load(os.path.join(C.ARTIFACTS_DIR, "oof_violation.npy"))

    # база v3 (LR)
    base = {n: cv_score(data, real, n, V3_FEATURES[n], V3_SOURCES[n], "lr", oof_cnn)
            for n in C.VIOLATIONS}
    mb, sb = honest(data, real, base, args.seeds)
    print(f"БАЗА v3 (LR)  macro-F1 {mb:.3f} ± {sb:.3f}\n")

    results = {"base": {"mean": mb, "std": sb}}
    for crit in ("femur_positioning", "spine_axis", "spine_artifacts", "femur_roi"):
        for kind in ("poly", "rf", "gbm", "svm"):
            S = dict(base)
            S[crit] = cv_score(data, real, crit, V3_FEATURES[crit], V3_SOURCES[crit], kind, oof_cnn)
            m, s = honest(data, real, S, args.seeds)
            results[f"{crit}:{kind}"] = {"mean": m, "std": s}
            print(f"{crit:18} {kind:5} macro-F1 {m:.3f} ± {s:.3f}  (Δ={m-mb:+.3f})")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nсохранено: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
