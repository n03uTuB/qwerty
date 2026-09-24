# -*- coding: utf-8 -*-
"""Фокус на femur_positioning (ротация): перебор источника/признаков, парно к v3.

femur_positioning — самая частая метка (36/150) и главный источник ложных
срабатываний «укладки» (precision 0.37). Проверяем, поднимает ли гибрид
(CNN+геометрия) или другие признаки точность.

Запуск:
    python dxa_qc_work/scripts/exp_femur.py --seeds 20
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import _common  # noqa: F401

from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402

from exp_sweep import CriterionScores  # noqa: E402
from exp_confirm import per_seed_macro  # noqa: E402
from build_v3 import V3_FEATURES, V3_SOURCES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(_common.OUT, "exp_femur.json"))
    args = ap.parse_args()
    n = args.seeds

    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    cs = CriterionScores(data, real)
    base_S = {k: cs.get(k, V3_FEATURES[k], V3_SOURCES[k]) for k in C.VIOLATIONS}
    base = per_seed_macro(data, real, base_S, n)
    print(f"БАЗА v3  {base.mean():.3f} ± {base.std():.3f}\n")

    MARG = ["femur_margin_min_cm", "femur_width_cm"]
    MARG3 = ["femur_margin_min_cm", "femur_margin_top_cm", "femur_margin_bottom_cm", "femur_width_cm"]
    TROCH = ["femur_trochanter_bulge", "femur_troch_area_mm2", "femur_neck_width_mm"]
    FEM3 = ["femur_trochanter_bulge", "femur_axis_angle", "femur_height_cm"]
    ALLF = ["femur_margin_min_cm", "femur_width_cm", "femur_trochanter_bulge",
            "femur_axis_angle", "femur_height_cm", "femur_shaft_deg", "femur_bone_ratio"]

    cands = [
        ("fused[marg]", MARG, "fused"),
        ("fused[marg3]", MARG3, "fused"),
        ("fused[troch]", TROCH, "fused"),
        ("fused[FEM3]", FEM3, "fused"),
        ("fused[all7]", ALLF, "fused"),
        ("geo[all7]", ALLF, "geo"),
        ("geo[marg3]", MARG3, "geo"),
    ]
    results = {"base": {"mean": float(base.mean()), "std": float(base.std())}}
    for title, cols, src in cands:
        S = dict(base_S)
        S["femur_positioning"] = cs.get("femur_positioning", cols, src)
        arr = per_seed_macro(data, real, S, n)
        d = arr - base
        wins = int((d > 1e-9).sum())
        results[title] = dict(mean=float(arr.mean()), std=float(arr.std()),
                              diff=float(d.mean()), wins=wins, n=n)
        print(f"{title:14} {arr.mean():.3f} ± {arr.std():.3f}  Δ={d.mean():+.3f}  wins {wins}/{n}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nсохранено: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
