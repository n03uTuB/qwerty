# -*- coding: utf-8 -*-
"""Парное подтверждение кандидатов на 20 разбиениях (знак разницы + доля выигрышей).

Кандидаты (по результатам exp_sweep):
  A) spine_axis: fused с добавлением spine_midline_residual (было +0.056 на 8 seeds);
  B) spine_artifacts: источник fused вместо cnn (было +0.009);
  C) A+B.

Парность: разбиения детерминированы seed'ом, поэтому для каждого seed считаем
базу и кандидата на ОДНИХ И ТЕХ ЖЕ фолдах и смотрим знак разницы.

Запуск:
    python dxa_qc_work/scripts/exp_confirm.py --seeds 20
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import _common  # noqa: F401

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402
from src.train import pick_threshold  # noqa: E402

from exp_sweep import BASE_FEATURES, BASE_SOURCES, CriterionScores, ORG_LABELS  # noqa: E402


def per_seed_macro(data, real, S_cols, n_seeds):
    """Список macro-F1 по разбиениям (порог по train-части фолда)."""
    idx = np.where(real)[0]
    g = data["study_uid"].values[idx]
    strat = data["region_idx"].values[idx]
    out = []
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
                p = S_cols[name]
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
        out.append(float(np.mean(fs)))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(_common.OUT, "exp_confirm.json"))
    args = ap.parse_args()
    n = args.seeds

    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    print(f"строк {len(data)} реальных {int(real.sum())} seeds={n}\n")

    cs = CriterionScores(data, real)
    base_S = {k: cs.get(k, BASE_FEATURES[k], BASE_SOURCES[k]) for k in C.VIOLATIONS}
    base = per_seed_macro(data, real, base_S, n)

    def variant(axis_cols=None, axis_src=None, art_src=None):
        S = dict(base_S)
        if axis_cols is not None:
            S["spine_axis"] = cs.get("spine_axis", axis_cols, axis_src or BASE_SOURCES["spine_axis"])
        if art_src is not None:
            S["spine_artifacts"] = cs.get("spine_artifacts", BASE_FEATURES["spine_artifacts"], art_src)
        return per_seed_macro(data, real, S, n)

    axis_res = ["spine_midline_angle", "spine_midline_residual"]
    cands = {
        "A) ось fused +residual": variant(axis_cols=axis_res),
        "A2) ось fused +res+axis_angle": variant(axis_cols=axis_res + ["spine_axis_angle"]),
        "A3) ось fused +res+iliac": variant(axis_cols=axis_res + ["spine_iliac_signal"]),
        "B) предметы fused": variant(art_src="fused"),
        "C) A+B": variant(axis_cols=axis_res, art_src="fused"),
    }

    print(f"БАЗА  macro-F1 {base.mean():.3f} ± {base.std():.3f}")
    results = {"base": {"mean": float(base.mean()), "std": float(base.std())}}
    for title, arr in cands.items():
        d = arr - base
        wins = int((d > 1e-9).sum())
        losses = int((d < -1e-9).sum())
        results[title] = dict(mean=float(arr.mean()), std=float(arr.std()),
                              diff=float(d.mean()), diff_std=float(d.std()),
                              wins=wins, losses=losses, n=n)
        print(f"{title:32} {arr.mean():.3f} ± {arr.std():.3f}   "
              f"Δ={d.mean():+.3f} ± {d.std():.3f}   выигрыш {wins}/{n}, проигрыш {losses}/{n}")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"\nсохранено: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
