# -*- coding: utf-8 -*-
"""Парный перебор кандидатов по одному критерию (20 разбиений, знак разницы).

Меняем один критерий за раз относительно базы v3, считаем Δmacro-F1 и число
выигрышных разбиений. Кандидаты с уверенным плюсом (>=16/20) идут в финал.

Запуск:
    python dxa_qc_work/scripts/exp_pairs.py
"""
from __future__ import annotations

import json
import os

import numpy as np

import _common  # noqa: F401

from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402

from exp_sweep import CriterionScores, ORG_LABELS  # noqa: E402
from exp_confirm import per_seed_macro  # noqa: E402
from build_v3 import V3_FEATURES, V3_SOURCES  # noqa: E402


def main():
    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    n = 20
    print(f"строк {len(data)} реальных {int(real.sum())} seeds={n}\n")

    cs = CriterionScores(data, real)
    base_S = {k: cs.get(k, V3_FEATURES[k], V3_SOURCES[k]) for k in C.VIOLATIONS}
    base = per_seed_macro(data, real, base_S, n)
    print(f"БАЗА v3  macro-F1 {base.mean():.3f} ± {base.std():.3f}\n")

    # (title, criterion, features, source)
    cands = [
        # --- укладка позвоночника (сейчас cnn) ---
        ("уклад fused", "spine_positioning", V3_FEATURES["spine_positioning"], "fused"),
        ("уклад +top_cut fused", "spine_positioning",
         ["spine_iliac_signal", "spine_bottom_cut", "spine_top_cut"], "fused"),
        ("уклад +margin fused", "spine_positioning",
         ["spine_iliac_signal", "spine_bottom_cut", "spine_margin_bottom_cm"], "fused"),
        ("уклад +all fused", "spine_positioning",
         ["spine_iliac_signal", "spine_bottom_cut", "spine_top_cut",
          "spine_margin_bottom_cm", "spine_height_cm"], "fused"),
        ("уклад +all geo", "spine_positioning",
         ["spine_iliac_signal", "spine_bottom_cut", "spine_top_cut",
          "spine_margin_bottom_cm", "spine_height_cm"], "geo"),
        # --- укладка бедра (сейчас geo) ---
        ("бедро fused", "femur_positioning", V3_FEATURES["femur_positioning"], "fused"),
        ("бедро +margin3 geo", "femur_positioning",
         ["femur_margin_min_cm", "femur_margin_top_cm", "femur_margin_bottom_cm",
          "femur_width_cm"], "geo"),
        ("бедро +shaft_w geo", "femur_positioning",
         ["femur_margin_min_cm", "femur_width_cm", "femur_shaft_width_cm"], "geo"),
        ("бедро +height geo", "femur_positioning",
         ["femur_margin_min_cm", "femur_width_cm", "femur_height_cm"], "geo"),
        ("бедро +axis_angle geo", "femur_positioning",
         ["femur_margin_min_cm", "femur_width_cm", "femur_axis_angle"], "geo"),
        ("бедро +bone_ratio geo", "femur_positioning",
         ["femur_margin_min_cm", "femur_width_cm", "femur_bone_ratio"], "geo"),
        # --- ROI (сейчас cnn), вдруг fused лучше ---
        ("ROI fused", "femur_roi", V3_FEATURES["femur_roi"], "fused"),
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
        print(f"{title:22} {arr.mean():.3f} ± {arr.std():.3f}  Δ={d.mean():+.3f}  "
              f"выигрыш {wins}/{n}")

    print("\n--- отсортировано ---")
    for title, r in sorted(results.items(), key=lambda kv: -kv[1]["mean"]):
        print(f"{title:22} {r['mean']:.3f}  Δ={r.get('diff', 0):+.3f}  wins={r.get('wins','-')}")

    out = os.path.join(_common.OUT, "exp_pairs.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nсохранено: {out}")


if __name__ == "__main__":
    main()
