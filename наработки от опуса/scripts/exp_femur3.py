# -*- coding: utf-8 -*-
"""P2b: конвертировать улучшенное ранжирование бедра (f2_*) в F1.

f2rot_norm(fused) даёт AUC 0.676 против 0.566 у базы, но F1 не растёт. Здесь
перебираем комбинации признаков чистой маски с развёрнутыми (отступы/вертел),
чтобы найти конфигурацию, где рост AUC переходит в рост macro-F1. Дополнительно
печатаем precision/recall объединённой метки «укладка».

Запуск:
    python dxa_qc_work/scripts/exp_femur3.py --seeds 20
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import _common  # noqa: F401

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score  # noqa: E402

from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402
from src.train import pick_threshold  # noqa: E402

from exp_sweep import CriterionScores  # noqa: E402
from exp_confirm import per_seed_macro  # noqa: E402
from build_v3 import V3_FEATURES, V3_SOURCES  # noqa: E402


def crit_auc(data, real, score, name):
    reg = "spine" if name.startswith("spine") else "femur"
    m = data["region"].str.contains(reg).values & real
    y = data["viol_" + name].values.astype(float)[m]
    p = score[m]
    ok = ~np.isnan(y) & ~np.isnan(p)
    y, p = y[ok].astype(int), p[ok]
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))


def ukladka_pr(data, real, S, thr_fe=None):
    """precision/recall объединённой метки «укладка» на всех реальных снимках."""
    idx = np.where(real)[0]
    gt = np.zeros(len(data), dtype=int)
    pr = np.zeros(len(data), dtype=int)
    for name in ("spine_positioning", "femur_positioning"):
        reg = "spine" if name.startswith("spine") else "femur"
        m = data["region"].str.contains(reg).values & real
        y = np.nan_to_num(data["viol_" + name].values.astype(float), nan=0.0).astype(int)
        gt[m] = y[m]
        t = thr_fe if (thr_fe is not None and name == "femur_positioning") \
            else pick_threshold(y[m], S[name][m], C.THRESHOLD_MODE)[0]
        pr[m & (S[name] >= t)] = 1
    tp = int(((gt == 1) & (pr == 1)).sum())
    fp = int(((gt == 0) & (pr == 1)).sum())
    fn = int(((gt == 1) & (pr == 0)).sum())
    return tp / max(tp + fp, 1), tp / max(tp + fn, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(_common.OUT, "exp_femur3.json"))
    args = ap.parse_args()
    n = args.seeds

    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    f2 = pd.read_csv(os.path.join(_common.OUT, "feat_femur2.csv")).drop_duplicates("image_uid")
    data = data.merge(f2, on="image_uid", how="left")
    real = np.asarray(ds.real_mask(data))
    print(f"строк {len(data)} реальных {int(real.sum())} seeds={n}\n")

    cs = CriterionScores(data, real)
    base_S = {k: cs.get(k, V3_FEATURES[k], V3_SOURCES[k]) for k in C.VIOLATIONS}
    base = per_seed_macro(data, real, base_S, n)
    b_auc, b_ap = crit_auc(data, real, base_S["femur_positioning"], "femur_positioning")
    p0, r0 = ukladka_pr(data, real, base_S)
    print(f"БАЗА v3  macro {base.mean():.3f}  бедро AUC={b_auc:.3f} AP={b_ap:.3f}  "
          f"| укладка P={p0:.3f} R={r0:.3f}")

    MARG = ["femur_margin_min_cm", "femur_width_cm"]
    NORM = ["f2_troch_area_norm", "f2_troch_peak_norm", "f2_neck_to_head", "f2_shaft_deg"]
    ROT = ["f2_troch_bulge", "f2_neck_width_mm", "f2_head_diameter_mm", "f2_shaft_deg",
           "f2_axis_angle"]
    BEST3 = ["f2_troch_area_norm", "f2_neck_width_mm", "f2_shaft_deg"]
    cands = {
        "norm+marg(fused)": (NORM + MARG, "fused"),
        "rot+norm(fused)": (ROT + NORM, "fused"),
        "rot+norm+marg(fused)": (ROT + NORM + MARG, "fused"),
        "best3(fused)": (BEST3, "fused"),
        "best3+marg(fused)": (BEST3 + MARG, "fused"),
        "norm(fused)": (NORM, "fused"),
        "norm+marg(geo)": (NORM + MARG, "geo"),
        "best3(geo)": (BEST3, "geo"),
        "best3+marg(geo)": (BEST3 + MARG, "geo"),
        "rot_norm+marg_geo(fused)": (NORM + ["f2_margin_min_cm", "f2_margin_top_cm"], "fused"),
    }
    results = {"base": dict(mean=float(base.mean()), std=float(base.std()), auc=b_auc, ap=b_ap)}
    for title, (cols_, src) in cands.items():
        S = dict(base_S)
        S["femur_positioning"] = cs.get("femur_positioning", cols_, src)
        arr = per_seed_macro(data, real, S, n)
        d = arr - base
        a, ap_ = crit_auc(data, real, S["femur_positioning"], "femur_positioning")
        p, r = ukladka_pr(data, real, S)
        wins = int((d > 1e-9).sum())
        results[title] = dict(mean=float(arr.mean()), std=float(arr.std()), diff=float(d.mean()),
                              wins=wins, n=n, auc=a, ap=ap_, prec=p, rec=r)
        print(f"{title:26} {arr.mean():.3f} Δ={d.mean():+.3f} wins {wins:2d}/{n}  "
              f"AUC={a:.3f} AP={ap_:.3f}  укладка P={p:.3f} R={r:.3f}")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"\nсохранено: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
