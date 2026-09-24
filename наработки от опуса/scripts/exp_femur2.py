# -*- coding: utf-8 -*-
"""P2-тест: признаки ротации на чистой маске бедра (feat_femur2.csv).

Сравнивает femur_positioning на новых признаках f2_* (отделение таза + нормировка
на головку) с развёрнутой конфигурацией v3. Парно по 20 разбиениям, с контролем
AUC критерия (требование Astra: рост ранжирования, а не только F1).

Запуск:
    python dxa_qc_work/scripts/exp_femur2.py --seeds 20
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import _common  # noqa: F401

from sklearn.metrics import roc_auc_score, average_precision_score  # noqa: E402

from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(_common.OUT, "exp_femur2.json"))
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
    print(f"БАЗА v3  macro-F1 {base.mean():.3f} ± {base.std():.3f}  "
          f"| бедро AUC={b_auc:.3f} AP={b_ap:.3f}")

    MARG = ["femur_margin_min_cm", "femur_width_cm"]
    ROT2 = ["f2_troch_bulge", "f2_neck_width_mm", "f2_head_diameter_mm", "f2_shaft_deg",
            "f2_axis_angle"]
    ROT2N = ["f2_troch_area_norm", "f2_troch_peak_norm", "f2_neck_to_head", "f2_shaft_deg"]
    ALL2 = ROT2 + ["f2_troch_area", "f2_margin_min_cm", "f2_bone_ratio", "f2_pelvis_ratio"]
    cands = {
        "бедро: f2rot(fused)": (ROT2, "fused"),
        "бедро: f2rot_norm(fused)": (ROT2N, "fused"),
        "бедро: f2all(fused)": (ALL2, "fused"),
        "бедро: f2rot+marg(fused)": (ROT2 + MARG, "fused"),
        "бедро: f2rot(geo)": (ROT2, "geo"),
        "бедро: f2rot+marg(geo)": (ROT2 + MARG, "geo"),
        "бедро: f2rot_norm(geo)": (ROT2N, "geo"),
    }
    results = {"base": dict(mean=float(base.mean()), std=float(base.std()), auc=b_auc, ap=b_ap)}
    for title, (cols_, src) in cands.items():
        S = dict(base_S)
        S["femur_positioning"] = cs.get("femur_positioning", cols_, src)
        arr = per_seed_macro(data, real, S, n)
        d = arr - base
        a, ap_ = crit_auc(data, real, S["femur_positioning"], "femur_positioning")
        wins = int((d > 1e-9).sum())
        results[title] = dict(mean=float(arr.mean()), std=float(arr.std()),
                              diff=float(d.mean()), wins=wins, n=n, auc=a, ap=ap_)
        print(f"{title:30} {arr.mean():.3f} Δ={d.mean():+.3f} wins {wins}/{n}  "
              f"| AUC={a:.3f} ({a-b_auc:+.3f}) AP={ap_:.3f} ({ap_-b_ap:+.3f})")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"\nсохранено: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
