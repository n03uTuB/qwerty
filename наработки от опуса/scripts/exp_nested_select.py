# -*- coding: utf-8 -*-
"""Честный отбор признаков: селекция ВНУТРИ train-части (nested), затем macro-F1.

Жадный отбор на полных данных даёт оптимистичные AUC. Здесь для каждого критерия
отбор делается только по train-части внешнего фолда (внутренний GroupKFold), а
оценка — на val. Так проверяем, реален ли выигрыш новых наборов признаков.

Запуск:
    python dxa_qc_work/scripts/exp_nested_select.py --seeds 10 --k 4
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import _common  # noqa: F401

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402
from src.train import pick_threshold  # noqa: E402

from exp_sweep import ORG_LABELS  # noqa: E402

FEATS = [f for f in fl.FEATURE_NAMES if f != "copies"]


def _clf():
    return Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(max_iter=2000, class_weight="balanced"))])


def inner_auc(X, y, g, cols_idx):
    Xs = X[:, cols_idx]
    p = np.full(len(y), np.nan)
    for tr, va in GroupKFold(n_splits=5).split(Xs, y, groups=g):
        if len(np.unique(y[tr])) < 2:
            continue
        p[va] = _clf().fit(Xs[tr], y[tr]).predict_proba(Xs[va])[:, 1]
    ok = ~np.isnan(p)
    if ok.sum() < 5 or len(np.unique(y[ok])) < 2:
        return -1.0
    return roc_auc_score(y[ok], p[ok])


def select_features(X, y, g, k):
    chosen, best = [], -1.0
    for _ in range(k):
        cand_i, cand_a = None, best
        for i in range(X.shape[1]):
            if i in chosen:
                continue
            a = inner_auc(X, y, g, chosen + [i])
            if a > cand_a + 1e-4:
                cand_a, cand_i = a, i
        if cand_i is None:
            break
        chosen.append(cand_i)
        best = cand_a
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(_common.OUT, "exp_nested_select.json"))
    args = ap.parse_args()
    n, K = args.seeds, args.k

    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    oof_cnn = np.load(os.path.join(C.ARTIFACTS_DIR, "oof_violation.npy"))
    idx = np.where(real)[0]
    g = data["study_uid"].values[idx]
    strat = data["region_idx"].values[idx]
    print(f"реальных {int(real.sum())} seeds={n} k={K}\n")

    macros = []
    for seed in range(n):
        splits = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                      random_state=seed).split(idx, strat, groups=g)
        fired = np.zeros((len(data), len(C.VIOLATIONS)), dtype=int)
        for tr_i, va_i in splits:
            tr, va = idx[tr_i], idx[va_i]
            for j, name in enumerate(C.VIOLATIONS):
                reg = "spine" if name.startswith("spine") else "femur"
                rmask = data["region"].str.contains(reg).values
                y = data["viol_" + name].values.astype(float)
                mtr = tr[rmask[tr] & ~np.isnan(y[tr])]
                mva = va[rmask[va] & ~np.isnan(y[va])]
                if len(mtr) < 10 or len(np.unique(y[mtr].astype(int))) < 2:
                    continue
                Xtr = data.loc[mtr, FEATS].fillna(0).values
                ytr = y[mtr].astype(int)
                gtr = data.loc[mtr, "study_uid"].values
                sel = select_features(Xtr, ytr, gtr, K)
                if not sel:
                    continue
                clf = _clf().fit(Xtr[:, sel], ytr)
                thr, _ = pick_threshold(ytr, clf.predict_proba(Xtr[:, sel])[:, 1], C.THRESHOLD_MODE)
                Xva = data.loc[mva, FEATS].fillna(0).values
                p = clf.predict_proba(Xva[:, sel])[:, 1]
                fired[mva[p >= thr], j] = 1
        fs = []
        for lab, crits in ORG_LABELS.items():
            gt = np.zeros(len(data), dtype=int)
            pr = np.zeros(len(data), dtype=int)
            for name in crits:
                gt |= np.nan_to_num(data["viol_" + name].values.astype(float), nan=0.0).astype(int)
                pr |= fired[:, C.VIOLATION_IDX[name]]
            fs.append(f1_score(gt[idx], pr[idx], zero_division=0))
        macros.append(float(np.mean(fs)))
        print(f"seed {seed}: macro-F1={macros[-1]:.3f}")

    print(f"\nЧЕСТНЫЙ ОТБОР (nested): macro-F1 {np.mean(macros):.3f} ± {np.std(macros):.3f}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"seeds": n, "k": K, "macro": float(np.mean(macros)),
                   "std": float(np.std(macros)), "per_seed": macros}, f, ensure_ascii=False, indent=2)
    print(f"сохранено: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
