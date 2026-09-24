# -*- coding: utf-8 -*-
"""Проверка новых наборов признаков (из жадного отбора) в честном протоколе.

Наборы выбраны на полных данных (жадный отбор), поэтому проверяем их честно:
порог по train-части фолда, парно с базой v3, 20 разбиений. Это защита от
оптимизма отбора признаков.

Запуск:
    python dxa_qc_work/scripts/exp_featsets.py --seeds 20
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
    ap.add_argument("--out", default=os.path.join(_common.OUT, "exp_featsets.json"))
    args = ap.parse_args()
    n = args.seeds

    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    print(f"строк {len(data)} реальных {int(real.sum())} seeds={n}\n")

    cs = CriterionScores(data, real)
    base_S = {k: cs.get(k, V3_FEATURES[k], V3_SOURCES[k]) for k in C.VIOLATIONS}
    base = per_seed_macro(data, real, base_S, n)
    print(f"БАЗА v3  macro-F1 {base.mean():.3f} ± {base.std():.3f}\n")

    AX5 = ["spine_midline_angle", "vertical_symmetry", "spine_midline_residual",
           "bone_eccentricity", "spine_iliac_signal"]
    AX3 = ["spine_midline_angle", "vertical_symmetry", "spine_midline_residual"]
    ART6 = ["spine_ribs_signal", "spine_vertebra_peaks", "spine_midline_residual",
            "vertical_symmetry", "spine_bone_ratio", "spine_height_cm"]
    ROI5 = ["femur_height_cm", "femur_head_diameter_mm", "femur_axis_angle",
            "bright_area_ratio", "bone_eccentricity"]
    FEM3 = ["femur_trochanter_bulge", "femur_axis_angle", "femur_height_cm"]

    cands = [
        ("ось geo5", "spine_axis", AX5, "geo"),
        ("ось geo3", "spine_axis", AX3, "geo"),
        ("ось fused5", "spine_axis", AX5, "fused"),
        ("пред geo6", "spine_artifacts", ART6, "geo"),
        ("пред fused6", "spine_artifacts", ART6, "fused"),
        ("ROI geo5", "femur_roi", ROI5, "geo"),
        ("ROI fused5", "femur_roi", ROI5, "fused"),
        ("бедро geo3", "femur_positioning", FEM3, "geo"),
        ("бедро fused3", "femur_positioning", FEM3, "fused"),
    ]

    results = {"base": {"mean": float(base.mean()), "std": float(base.std())}}
    for title, crit, cols, src in cands:
        S = dict(base_S)
        S[crit] = cs.get(crit, cols, src)
        arr = per_seed_macro(data, real, S, n)
        d = arr - base
        wins = int((d > 1e-9).sum())
        results[title] = dict(mean=float(arr.mean()), std=float(arr.std()),
                              diff=float(d.mean()), wins=wins, n=n,
                              crit=crit, cols=cols, src=src)
        print(f"{title:14} {arr.mean():.3f} ± {arr.std():.3f}  Δ={d.mean():+.3f}  wins {wins}/{n}")

    # комбинация лучших (по одному критерию)
    print("\n--- комбинация победителей ---")
    best = {c: (V3_FEATURES[c], V3_SOURCES[c]) for c in C.VIOLATIONS}
    for crit, cols, src in [("spine_axis", AX5, "geo"), ("spine_artifacts", ART6, "fused"),
                            ("femur_roi", ROI5, "geo"), ("femur_positioning", FEM3, "geo")]:
        # тестируем добавление каждого к накопленному лучшему
        pass
    # простой вариант: все четыре замены сразу
    S = dict(base_S)
    for crit, cols, src in [("spine_axis", AX5, "geo"), ("spine_artifacts", ART6, "fused"),
                            ("femur_roi", ROI5, "geo"), ("femur_positioning", FEM3, "geo")]:
        S[crit] = cs.get(crit, cols, src)
    arr = per_seed_macro(data, real, S, n)
    d = arr - base
    results["ALL4"] = dict(mean=float(arr.mean()), std=float(arr.std()),
                           diff=float(d.mean()), wins=int((d > 1e-9).sum()), n=n)
    print(f"{'ALL4':14} {arr.mean():.3f} ± {arr.std():.3f}  Δ={d.mean():+.3f}  "
          f"wins {int((d>1e-9).sum())}/{n}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nсохранено: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
