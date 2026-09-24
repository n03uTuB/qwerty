# -*- coding: utf-8 -*-
"""P1: прямые измерения «посторонних предметов» для spine_artifacts (идея Astra).

Проверяет, поднимают ли признаки foreign_* (посчитанные feat_foreign.py) качество
критерия spine_artifacts в боевой конфигурации dxa_qc (fused-стекер CNN+геометрия).
Требование Astra: растёт именно AUC/AP критерия, а не только macro-F1.

Запуск:
    python dxa_qc_work/scripts/exp_artifacts.py --seeds 20
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
    """AUC/AP критерия по реальным снимкам его области."""
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
    ap.add_argument("--out", default=os.path.join(_common.OUT, "exp_artifacts.json"))
    args = ap.parse_args()
    n = args.seeds

    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    fcsv = os.path.join(_common.OUT, "feat_foreign.csv")
    fdf = pd.read_csv(fcsv).drop_duplicates("image_uid", keep="first")
    cols = [c for c in fdf.columns if c != "image_uid"]
    data = data.merge(fdf, on="image_uid", how="left")
    real = np.asarray(ds.real_mask(data))
    print(f"строк {len(data)} реальных {int(real.sum())} seeds={n}\n")

    cs = CriterionScores(data, real)
    base_S = {k: cs.get(k, V3_FEATURES[k], V3_SOURCES[k]) for k in C.VIOLATIONS}
    base = per_seed_macro(data, real, base_S, n)
    b_auc, b_ap = crit_auc(data, real, base_S["spine_artifacts"], "spine_artifacts")
    print(f"БАЗА v3  macro-F1 {base.mean():.3f} ± {base.std():.3f}  "
          f"| предметы AUC={b_auc:.3f} AP={b_ap:.3f}")

    RIB = ["spine_ribs_signal", "spine_vertebra_peaks"]
    FOREIGN = ["foreign_count", "foreign_area", "foreign_max_area",
               "foreign_compactness", "foreign_top"]
    FOREIGN_NEW = ["foreign_peak_rel", "foreign_asym", "foreign_elong_max",
                   "foreign_elong_count", "foreign_area_left", "foreign_area_right"]
    cands = {
        "пред: ribs+foreign(fused)": (RIB + FOREIGN, "fused"),
        "пред: ribs+foreign(new)(fused)": (RIB + FOREIGN_NEW, "fused"),
        "пред: ribs+all foreign(fused)": (RIB + FOREIGN + FOREIGN_NEW, "fused"),
        "пред: foreign only(fused)": (FOREIGN + FOREIGN_NEW, "fused"),
        "пред: foreign only(geo)": (FOREIGN + FOREIGN_NEW, "geo"),
        "пред: ribs+foreign(geo)": (RIB + FOREIGN, "geo"),
    }
    results = {"base": dict(mean=float(base.mean()), std=float(base.std()),
                            auc=b_auc, ap=b_ap)}
    for title, (cols_, src) in cands.items():
        S = dict(base_S)
        S["spine_artifacts"] = cs.get("spine_artifacts", cols_, src)
        arr = per_seed_macro(data, real, S, n)
        d = arr - base
        a, ap_ = crit_auc(data, real, S["spine_artifacts"], "spine_artifacts")
        wins = int((d > 1e-9).sum())
        results[title] = dict(mean=float(arr.mean()), std=float(arr.std()),
                              diff=float(d.mean()), wins=wins, n=n, auc=a, ap=ap_)
        print(f"{title:34} {arr.mean():.3f} Δ={d.mean():+.3f} wins {wins}/{n}  "
              f"| AUC={a:.3f} ({a-b_auc:+.3f}) AP={ap_:.3f} ({ap_-b_ap:+.3f})")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"\nсохранено: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
