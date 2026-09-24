# -*- coding: utf-8 -*-
"""P2c: конвертировать рост AUC бедра в F1 через правило порога.

f2-скор бедра даёт AUC 0.68 против 0.57, но развёрнутое правило порога (max F1 по
train, сетка) не переносит это в macro-F1. Проверяем альтернативные ЧЕСТНЫЕ правила
порога (всё подбирается на train-части фолда) для базового и нового скора бедра.

Запуск:
    python dxa_qc_work/scripts/exp_femur_thr.py --seeds 20
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import _common  # noqa: F401

from sklearn.metrics import f1_score, roc_auc_score, average_precision_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402
from src.train import pick_threshold  # noqa: E402

from exp_sweep import CriterionScores, ORG_LABELS  # noqa: E402
from build_v3 import V3_FEATURES, V3_SOURCES  # noqa: E402
from exp_thresh import THR  # noqa: E402

NORM = ["f2_troch_area_norm", "f2_troch_peak_norm", "f2_neck_to_head", "f2_shaft_deg"]


def eval_with(data, real, S, thr_fn, n_seeds):
    """macro-F1, где порог ВСЕХ критериев берётся заданной функцией thr_fn.

    thr_fn(criterion_name, y, p) -> threshold.
    """
    idx = np.where(real)[0]
    g = data["study_uid"].values[idx]
    strat = data["region_idx"].values[idx]
    macros, per = [], {k: [] for k in ORG_LABELS}
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
                thr = thr_fn(name, y[trc].astype(int), p[trc])
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
            per[lab].append(f)
            fs.append(f)
        macros.append(float(np.mean(fs)))
    return float(np.mean(macros)), float(np.std(macros)), {k: float(np.mean(v)) for k, v in per.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(_common.OUT, "exp_femur_thr.json"))
    args = ap.parse_args()
    n = args.seeds

    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    f2 = pd.read_csv(os.path.join(_common.OUT, "feat_femur2.csv")).drop_duplicates("image_uid")
    data = data.merge(f2, on="image_uid", how="left")
    real = np.asarray(ds.real_mask(data))

    cs = CriterionScores(data, real)
    base_S = {k: cs.get(k, V3_FEATURES[k], V3_SOURCES[k]) for k in C.VIOLATIONS}
    norm_S = dict(base_S)
    norm_S["femur_positioning"] = cs.get("femur_positioning", NORM, "fused")

    rules = {
        "f1(сетка)": lambda name, y, p: pick_threshold(y, p, "f1")[0],
        "prior": lambda name, y, p: THR["prior"](y, p),
        "youden": lambda name, y, p: THR["youden"](y, p),
        "ap": lambda name, y, p: THR["ap"](y, p),
        "median_pos": lambda name, y, p: float(np.median(p[y == 1])) if (y == 1).any() else 0.5,
        "mid_class": lambda name, y, p: float((np.median(p[y == 1]) + np.median(p[y == 0])) / 2)
        if (y == 1).any() and (y == 0).any() else 0.5,
    }

    # Варьируем правило ТОЛЬКО для femur_positioning; остальные критерии — базовый f1.
    def make(rule):
        def fn(name, y, p):
            if name == "femur_positioning":
                return rule(name, y, p)
            return pick_threshold(y, p, "f1")[0]
        return fn

    results = {}
    for sname, S in (("base", base_S), ("norm", norm_S)):
        results[sname] = {}
        for rname, rule in rules.items():
            m, s, pl = eval_with(data, real, S, make(rule), n)
            results[sname][rname] = dict(macro=m, std=s, per_label=pl)
            print(f"[{sname:4}] femur thr={rname:11} macro={m:.3f} ± {s:.3f}  "
                  + " ".join(f"{k}={v:.3f}" for k, v in pl.items()))

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"\nсохранено: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
