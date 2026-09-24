# -*- coding: utf-8 -*-
"""Решающий слой: стратегии порога и ансамблирование скоров (честный протокол).

Мотив (находки репозитория): главное узкое место — не измерение, а ВЫБОР ПОРОГА
при 6–36 положительных. AUC оси 0.82, а F1 0.42. Проверяем на 20 разбиениях,
порог подбирается только по train-части фолда:

  1. стратегии: f1, prior (доля нарушений), prior*k, youden, ap (максимум AP);
  2. ансамбль рангов cnn+geo;
  3. выбор лучшей стратегии на каждую метку организатора.

Запуск:
    python dxa_qc_work/scripts/exp_thresh.py --seeds 20
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import _common  # noqa: F401

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402
from src import stack as st        # noqa: E402

ORG_LABELS = {
    "укладка": ["spine_positioning", "femur_positioning"],
    "ось": ["spine_axis"],
    "предметы": ["spine_artifacts"],
    "ROI": ["femur_roi"],
}

# Конфигурация v3 (residual оси + предметы fused) — решающий слой тестируем на ней.
from build_v3 import V3_FEATURES as BASE_FEATURES, V3_SOURCES as BASE_SOURCES  # noqa: E402


def _clf():
    return Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))])


def _cv(X, y, groups, cnn=None):
    oof = np.full(len(y), np.nan)
    if len(np.unique(y)) < 2:
        return oof
    Xf = X if cnn is None else np.column_stack([cnn, X])
    for tr, va in GroupKFold(n_splits=5).split(Xf, y, groups=groups):
        if len(np.unique(y[tr])) < 2:
            continue
        oof[va] = _clf().fit(Xf[tr], y[tr]).predict_proba(Xf[va])[:, 1]
    return oof


def _ranks(p):
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    if ok.sum() == 0:
        return out
    order = np.argsort(np.argsort(p[ok]))
    out[ok] = order / max(len(order) - 1, 1)
    return out


class Scores:
    def __init__(self, data, real):
        self.data, self.real = data, real
        self.cnn = np.load(os.path.join(C.ARTIFACTS_DIR, "oof_violation.npy"))
        self.cache = {}

    def get(self, name, cols, source):
        key = (name, tuple(cols), source)
        if key in self.cache:
            return self.cache[key]
        j = C.VIOLATION_IDX[name]
        reg = "spine" if name.startswith("spine") else "femur"
        rmask = self.data["region"].str.contains(reg).values & self.real
        y = self.data["viol_" + name].values.astype(float)
        known = ~np.isnan(y) & rmask
        out = np.full(len(self.data), np.nan)
        cnn = self.cnn[known, j]
        if source == "cnn":
            out[known] = cnn
        else:
            cols = [c for c in cols if c in self.data.columns]
            X = self.data.loc[known, cols].fillna(0).values if cols else np.zeros((known.sum(), 1))
            yk = y[known].astype(int)
            gk = self.data.loc[known, "study_uid"].values
            out[known] = _cv(X, yk, gk, cnn=(cnn if source == "fused" else None))
        self.cache[key] = out
        return out

    def ens(self, name, cols):
        a = _ranks(self.get(name, cols, "fused"))
        b = _ranks(self.get(name, [], "cnn"))
        return np.nanmean(np.vstack([a, b]), axis=0)


# --------------------------- стратегии порога --------------------------- #
def thr_f1(y, p):
    ths = np.unique(p)
    best, bt = -1.0, float(np.median(p))
    for t in ths:
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best:
            best, bt = f, float(t)
    return bt


def thr_prior(y, p, k=1.0):
    prev = float(np.mean(y))
    if not (0 < prev < 1) or len(p) == 0:
        return float(np.median(p))
    cnt = max(1, int(round(prev * len(p) * k)))
    return float(np.sort(p)[::-1][min(cnt, len(p)) - 1])


def thr_youden(y, p):
    fpr, tpr, th = roc_curve(y, p)
    return float(th[int(np.argmax(tpr - fpr))])


def thr_ap(y, p):
    ths = np.unique(p)
    best, bt = -1.0, float(np.median(p))
    for t in ths:
        pred = (p >= t).astype(int)
        if 0 < pred.sum() < len(p):
            a = average_precision_score(y, pred)
            if a > best:
                best, bt = a, float(t)
    return bt


THR = {
    "f1": thr_f1,
    "prior": thr_prior,
    "prior1.2": lambda y, p: thr_prior(y, p, 1.2),
    "prior0.8": lambda y, p: thr_prior(y, p, 0.8),
    "youden": thr_youden,
    "ap": thr_ap,
}


def eval_strategy(data, real, scores, modes, n_seeds=20):
    idx = np.where(real)[0]
    g = data["study_uid"].values[idx]
    strat = data["region_idx"].values[idx]
    macros, per_label = [], {k: [] for k in ORG_LABELS}
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
                p = scores[name]
                trc = tr[rmask[tr] & ~np.isnan(y[tr]) & ~np.isnan(p[tr])]
                if len(trc) < 2 or len(np.unique(y[trc].astype(int))) < 2:
                    continue
                thr = modes[name](y[trc].astype(int), p[trc])
                va_ok = va[~np.isnan(p[va])]
                fired[va_ok[p[va_ok] >= thr], j] = 1
        fs = []
        for lab, crits in ORG_LABELS.items():
            gt = np.zeros(len(data), dtype=int)
            pr = np.zeros(len(data), dtype=int)
            for name in crits:
                gt |= np.nan_to_num(data["viol_" + name].values.astype(float), nan=0.0).astype(int)
                pr |= fired[:, C.VIOLATION_IDX[name]]
            f = f1_score(gt[idx], pr[idx], zero_division=0)
            per_label[lab].append(f)
            fs.append(f)
        macros.append(float(np.mean(fs)))
    return dict(macro=float(np.mean(macros)), macro_std=float(np.std(macros)),
                per_label={k: float(np.mean(v)) for k, v in per_label.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(_common.OUT, "exp_thresh.json"))
    args = ap.parse_args()

    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    print(f"строк {len(data)} реальных {int(real.sum())} seeds={args.seeds}\n")

    sc = Scores(data, real)
    S = {n: sc.get(n, BASE_FEATURES[n], BASE_SOURCES[n]) for n in C.VIOLATIONS}

    results = {}
    print("--- единая стратегия для всех критериев ---")
    for mode_name, fn in THR.items():
        modes = {n: fn for n in C.VIOLATIONS}
        r = eval_strategy(data, real, S, modes, n_seeds=args.seeds)
        results[f"mode={mode_name}"] = r
        print(f"mode={mode_name:10} macro-F1={r['macro']:.3f}  "
              + " ".join(f"{k}={v:.3f}" for k, v in r["per_label"].items()))

    print("\n--- лучшая стратегия на каждую метку (честные фолды) ---")
    best_per_label = {}
    per_mode_cache = {}
    for mode_name, fn in THR.items():
        per_mode_cache[mode_name] = eval_strategy(
            data, real, S, {n: fn for n in C.VIOLATIONS}, n_seeds=args.seeds)
    for lab, crits in ORG_LABELS.items():
        rows = sorted(((per_mode_cache[m]["per_label"][lab], m) for m in THR), reverse=True)
        best_per_label[lab] = rows[0][1]
        print(f"  {lab:10} лучший={rows[0][1]:10} {rows[0][0]:.3f}   "
              + " ".join(f"{m}={v:.3f}" for v, m in rows))

    comb_modes = {}
    for lab, crits in ORG_LABELS.items():
        for name in crits:
            comb_modes[name] = THR[best_per_label[lab]]
    r = eval_strategy(data, real, S, comb_modes, n_seeds=args.seeds)
    results["comb_best_per_label"] = r
    print(f"  ИТОГ комбинированный macro-F1={r['macro']:.3f}  "
          + " ".join(f"{k}={v:.3f}" for k, v in r["per_label"].items()))

    print("\n--- ансамбль рангов (fused+cnn) ---")
    S_ens = {n: sc.ens(n, BASE_FEATURES[n]) for n in C.VIOLATIONS}
    for mode_name, fn in THR.items():
        modes = {n: fn for n in C.VIOLATIONS}
        r = eval_strategy(data, real, S_ens, modes, n_seeds=args.seeds)
        results[f"ens:mode={mode_name}"] = r
        print(f"ens mode={mode_name:10} macro-F1={r['macro']:.3f}  "
              + " ".join(f"{k}={v:.3f}" for k, v in r["per_label"].items()))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"seeds": args.seeds, "results": results,
                   "best_per_label": best_per_label}, fh, ensure_ascii=False, indent=2)
    print(f"\nсохранено: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
