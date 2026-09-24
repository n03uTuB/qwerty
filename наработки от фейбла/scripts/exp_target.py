# -*- coding: utf-8 -*-
"""Стенд: честная оценка целевых критериев (ротация бедра, spine_artifacts).

Кэширует таблицу признаков один раз, затем быстро прогоняет варианты наборов
признаков по честной схеме (StratifiedGroupKFold по study_uid, вложенный порог,
оценка только по реальным снимкам — как dxa_real/evaluate).

Запуск:
    python scripts/exp_target.py --root <dataset_root> [--rebuild]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dxa_real import evaluate as ev          # noqa: E402
from dxa_real import features as F           # noqa: E402
from dxa_real.data import (LABEL_AXIS, LABEL_FOREIGN, LABEL_POSITIONING,  # noqa: E402
                           LABEL_ROI, load_dataset)

CACHE = os.path.join("out", "feat_cache.csv")

# Признаки, для которых строим разницу лево-право внутри исследования (находка 5).
LR_BASE = [
    "femur_margin_min_cm", "femur_margin_top_cm", "femur_margin_bottom_cm",
    "femur_bone_ratio", "femur_shaft_deg", "femur_trochanter_bulge",
    "femur_troch_area_mm2", "femur_neck_width_mm", "femur_height_cm",
    "femur_troch_peak_mm", "femur_head_diameter_mm", "femur_neck_to_head",
]


def build_cache(root: str) -> pd.DataFrame:
    images = load_dataset(root)
    rows = []
    for im in images:
        feat = F.compute(im)
        feat["study_uid"] = im.study_uid
        feat["region"] = im.region
        feat["side"] = im.side or ""
        feat["image_uid"] = im.image_uid
        rows.append(feat)
    table = pd.DataFrame(rows)
    # метки: качество и критерии ТЗ (как ev.build_table)
    table["quality"] = [im.quality for im in images]
    table["synthetic"] = [getattr(im, "synthetic", False) for im in images]
    from dxa_real.data import LABELS
    for label in LABELS:
        table[label] = [im.labels.get(label, np.nan) for im in images]

    # --- разница лево-право по исследованию (только бедро) ---
    for base in LR_BASE:
        table["dlr_" + base] = np.nan
    n_pairs = 0
    for uid, grp in table[table["region"] == "femur"].groupby("study_uid"):
        if len(grp) != 2:
            continue
        sides = set(grp["side"])
        if not {"l", "r"}.issubset(sides):
            continue
        idx_l = grp.index[grp["side"] == "l"][0]
        idx_r = grp.index[grp["side"] == "r"][0]
        n_pairs += 1
        for base in LR_BASE:
            if base in table.columns:
                d = abs(float(table.at[idx_l, base]) - float(table.at[idx_r, base]))
                table.at[idx_l, "dlr_" + base] = d
                table.at[idx_r, "dlr_" + base] = d
    print(f"[cache] исследований с парой бёдер: {n_pairs}")
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    table.to_csv(CACHE, index=False, encoding="utf-8")
    return table


def eval_set(table: pd.DataFrame, region: str, label: str, cols: list[str],
             threshold_mode: str = "nested") -> ev.Score:
    m = (table["region"] == region).values
    X = table.loc[m, cols].fillna(0.0).values.astype(float)
    y = table.loc[m, label].values.astype(float)
    known = ~np.isnan(y)
    if known.sum() == 0 or len(np.unique(y[known])) < 2:
        raise ValueError(f"нет данных для {region}/{label}")
    X, y, groups = X[known], y[known].astype(int), table.loc[m, "study_uid"].values[known]
    probs, preds = ev.cross_validate(X, y, groups, threshold_mode=threshold_mode)
    ok = ~np.isnan(probs)
    return ev.score(label, y[ok], probs[ok], preds[ok], groups[ok])


def show(name: str, s: ev.Score) -> None:
    lo, hi = s.auc_ci
    print(f"  {name:38} AUC={s.auc:.3f} [{lo:.2f};{hi:.2f}]  AP={s.ap:.3f}  F1={s.f1:.3f}  "
          f"ч={s.sensitivity:.2f} с={s.specificity:.2f}")


def univariate(table: pd.DataFrame, region: str, label: str, features: list[str]) -> None:
    """Одномерный AUC каждого признака по метке (только реальные снимки)."""
    from sklearn.metrics import roc_auc_score

    m = (table["region"] == region).values
    y = table.loc[m, label].values.astype(float)
    ok = ~np.isnan(y)
    y = y[ok].astype(int)
    if len(np.unique(y)) < 2:
        print("  нет позитивов")
        return
    scores = []
    for f in features:
        if f not in table.columns:
            continue
        x = table.loc[m, f].values[ok].astype(float)
        if np.all(np.isnan(x)) or np.nanstd(x) < 1e-12:
            continue
        x = np.nan_to_num(x)
        try:
            a = roc_auc_score(y, x)
            scores.append((max(a, 1 - a), a, f))
        except Exception:
            continue
    scores.sort(reverse=True)
    for _, a, f in scores:
        print(f"    {f:32} AUC={a:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--mode", default="nested", choices=["nested", "prior", "blend"])
    args = ap.parse_args()

    if args.rebuild or not os.path.isfile(CACHE):
        t0 = time.time()
        table = build_cache(args.root)
        print(f"[cache] посчитано за {time.time()-t0:.0f}s -> {CACHE}")
    else:
        table = pd.read_csv(CACHE)
        print(f"[cache] загружено {CACHE}: {len(table)} строк")

    print(f"\n=== РОТАЦИЯ БЕДРА (femur_positioning, n=36)  режим={args.mode} ===")
    femur_sets = {
        "V3 (baseline)": ["femur_trochanter_bulge", "femur_troch_area_mm2", "femur_neck_width_mm"],
        "V3 + bulge^2": ["femur_trochanter_bulge", "femur_troch_area_mm2",
                         "femur_neck_width_mm", "femur_trochanter_sq"],
        "bulge^2 + area + neck": ["femur_trochanter_sq", "femur_troch_area_mm2",
                                  "femur_neck_width_mm"],
        "V3 + LR-разницы": ["femur_trochanter_bulge", "femur_troch_area_mm2",
                            "femur_neck_width_mm", "dlr_femur_trochanter_bulge",
                            "dlr_femur_troch_area_mm2", "dlr_femur_neck_width_mm"],
        "только LR-разницы": ["dlr_femur_trochanter_bulge", "dlr_femur_troch_area_mm2",
                              "dlr_femur_neck_width_mm"],
        "LR(3 ротация) + LR(margin)": ["dlr_femur_trochanter_bulge", "dlr_femur_troch_area_mm2",
                                       "dlr_femur_neck_width_mm", "dlr_femur_margin_min_cm",
                                       "dlr_femur_bone_ratio", "dlr_femur_shaft_deg"],
        "V3 + LR ротация + neck/head": ["femur_trochanter_bulge", "femur_troch_area_mm2",
                                        "femur_neck_width_mm", "dlr_femur_trochanter_bulge",
                                        "dlr_femur_neck_width_mm", "femur_neck_to_head"],
        "broad (много)": ["femur_trochanter_bulge", "femur_troch_area_mm2", "femur_neck_width_mm",
                          "femur_trochanter_sq", "femur_shaft_deg", "femur_troch_peak_mm",
                          "femur_neck_to_head", "femur_margin_min_cm"],
    }
    base_auc = None
    for name, cols in femur_sets.items():
        try:
            s = eval_set(table, "femur", LABEL_POSITIONING, cols, args.mode)
            show(name, s)
            if name == "V3 (baseline)":
                base_auc = s.auc
        except Exception as e:
            print(f"  {name:38} ОШИБКА: {e}")

    print(f"\n=== SPINE_ARTIFACTS (n=17)  режим={args.mode} ===")
    art_sets = {
        "V3 (baseline)": ["spine_ribs_signal", "spine_vertebra_peaks"],
        "foreign_* (не использовались)": ["foreign_count", "foreign_area", "foreign_max_area",
                                          "foreign_compactness", "foreign_top"],
        "foreign_count + area": ["foreign_count", "foreign_area"],
        "foreign_max_area + compactness": ["foreign_max_area", "foreign_compactness"],
        "V3 + foreign_*": ["spine_ribs_signal", "spine_vertebra_peaks", "foreign_count",
                           "foreign_area", "foreign_max_area", "foreign_compactness",
                           "foreign_top"],
        "foreign + ribs": ["foreign_count", "foreign_area", "spine_ribs_signal"],
        "spine_ecc/bone_ratio/mean/std": ["spine_ecc", "spine_bone_ratio", "mean", "std",
                                          "spine_midline_residual"],
    }
    for name, cols in art_sets.items():
        try:
            s = eval_set(table, "spine", LABEL_FOREIGN, cols, args.mode)
            show(name, s)
        except Exception as e:
            print(f"  {name:38} ОШИБКА: {e}")

    print("\n=== РАУНД 2: комбинации по одномерному сигналу ===")
    femur_sets2 = {
        "shaft_deg + troch_position": ["femur_shaft_deg", "femur_troch_position"],
        "shaft_deg + troch_pos + bone_ratio": ["femur_shaft_deg", "femur_troch_position",
                                               "femur_bone_ratio"],
        "shaft_deg + troch_pos + margin_lat": ["femur_shaft_deg", "femur_troch_position",
                                               "femur_margin_lateral_cm"],
        "top5": ["femur_shaft_deg", "femur_troch_position", "femur_margin_lateral_cm",
                 "femur_bone_ratio", "femur_troch_peak_mm"],
        "top6 + head_diam + ischium": ["femur_shaft_deg", "femur_troch_position",
                                       "femur_margin_lateral_cm", "femur_bone_ratio",
                                       "femur_troch_peak_mm", "femur_head_diameter_mm",
                                       "femur_ischium_signal"],
        "dlr_bone_ratio + shaft_deg": ["dlr_femur_bone_ratio", "femur_shaft_deg"],
        "dlr_bone_ratio + shaft_deg + troch_pos": ["dlr_femur_bone_ratio", "femur_shaft_deg",
                                                   "femur_troch_position"],
        "V3 + shaft_deg + troch_pos": ["femur_trochanter_bulge", "femur_troch_area_mm2",
                                       "femur_neck_width_mm", "femur_shaft_deg",
                                       "femur_troch_position"],
        "shaft_deg_sq + troch_pos + margin_lat": ["femur_shaft_deg_sq", "femur_troch_position",
                                                  "femur_margin_lateral_cm"],
    }
    for name, cols in femur_sets2.items():
        try:
            s = eval_set(table, "femur", LABEL_POSITIONING, cols, args.mode)
            show(name, s)
        except Exception as e:
            print(f"  {name:38} ОШИБКА: {e}")

    art_sets2 = {
        "ribs alone": ["spine_ribs_signal"],
        "peaks alone": ["spine_vertebra_peaks"],
        "ribs + peaks (V3)": ["spine_ribs_signal", "spine_vertebra_peaks"],
        "V3 + midline_residual": ["spine_ribs_signal", "spine_vertebra_peaks",
                                  "spine_midline_residual"],
        "ribs + peaks + residual + ecc": ["spine_ribs_signal", "spine_vertebra_peaks",
                                          "spine_midline_residual", "spine_ecc"],
        "ribs + residual": ["spine_ribs_signal", "spine_midline_residual"],
    }
    for name, cols in art_sets2.items():
        try:
            s = eval_set(table, "spine", LABEL_FOREIGN, cols, args.mode)
            show(name, s)
        except Exception as e:
            print(f"  {name:38} ОШИБКА: {e}")

    print("\n=== ОДНОМЕРНЫЙ AUC: ротация бедра (все признаки бедра) ===")
    femur_all = [c for c in table.columns if c.startswith(("femur_", "dlr_"))]
    univariate(table, "femur", LABEL_POSITIONING, femur_all)

    print("\n=== ОДНОМЕРНЫЙ AUC: spine_artifacts (все признаки позвоночника) ===")
    spine_all = [c for c in table.columns if c.startswith(("spine_", "foreign_"))]
    univariate(table, "spine", LABEL_FOREIGN, spine_all)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
