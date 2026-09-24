# -*- coding: utf-8 -*-
"""P5: диагностика потолка femur_positioning (идея Astra).

1) PR-кривая критерия и precision при фиксированном recall 0.6/0.7/0.8;
2) гипотеза «ложные срабатывания = слияние маски с тазом»: корреляция FP с
   шириной шейки (>40 мм), с числом крупных связных компонент кости, с высотой;
3) какой одиночный признак/группа даёт лучший AUC — понять потолок измерения.

Запуск:
    python dxa_qc_work/scripts/diag_femur.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _common  # noqa: F401

from sklearn.metrics import roc_auc_score, average_precision_score  # noqa: E402

from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402


def main():
    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    m = data["region"].str.contains("femur").values & real
    y = data["viol_femur_positioning"].values.astype(float)[m]
    ok = ~np.isnan(y)
    y = y[ok].astype(int)
    sub = data[m][ok]
    print(f"femur_positioning: n={len(y)} pos={int(y.sum())} ({y.mean():.1%})\n")

    cand = {
        "femur_margin_min_cm": ["femur_margin_min_cm"],
        "femur_width_cm": ["femur_width_cm"],
        "margins(2)": ["femur_margin_min_cm", "femur_width_cm"],
        "troch(3)": ["femur_trochanter_bulge", "femur_troch_area_mm2", "femur_neck_width_mm"],
        "neck+head": ["femur_neck_width_mm", "femur_head_diameter_mm", "femur_neck_to_head"],
        "axis+shaft": ["femur_axis_angle", "femur_shaft_deg"],
        "all": [c for c in fl.FEATURE_NAMES if c.startswith("femur")],
    }
    print("AUC/AP одиночных признаков и групп (без CV, верхняя граница):")
    for title, cols in cand.items():
        cols = [c for c in cols if c in sub.columns]
        if not cols:
            continue
        X = sub[cols].fillna(0).values
        # простая нормировка и логистическая оценка через один признак/сумму z
        z = (X - X.mean(0)) / (X.std(0) + 1e-9)
        s = z.mean(1)
        auc = roc_auc_score(y, s)
        ap = average_precision_score(y, s)
        print(f"  {title:22} AUC={auc:.3f} AP={ap:.3f}")

    # PR при фиксированном recall для лучшей группы (margins+neck)
    cols = [c for c in ["femur_margin_min_cm", "femur_width_cm", "femur_neck_width_mm",
                        "femur_head_diameter_mm", "femur_axis_angle"] if c in sub.columns]
    X = sub[cols].fillna(0).values
    z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    s = z.mean(1)
    order = np.argsort(-s)
    ys = y[order]
    cum = np.cumsum(ys)
    for target in (0.6, 0.7, 0.8):
        need = int(np.ceil(target * y.sum()))
        if need <= 0 or need > len(ys):
            continue
        k = int(np.argmax(cum >= need)) + 1
        prec = cum[k - 1] / k
        print(f"  recall>={target:.1f}: нужно {k} снимков, precision={prec:.3f}")

    print("\nПроверка гипотезы «FP = слияние с тазом»:")
    thr = np.quantile(s, 1 - 0.55)  # примерно как развёрнутый порог
    pred = (s >= thr).astype(int)
    fp = (pred == 1) & (y == 0)
    for col in ["femur_neck_width_mm", "femur_head_diameter_mm", "femur_height_cm",
                "femur_bone_ratio", "femur_axis_angle"]:
        if col not in sub.columns:
            continue
        a = sub[col].fillna(0).values
        print(f"  {col:24} FP_mean={a[fp].mean():8.2f}  TP_mean={a[(pred==1)&(y==1)].mean():8.2f}"
              f"  TN_mean={a[(pred==0)&(y==0)].mean():8.2f}")

    # сколько компонент кости (слияние с тазом даёт крупную вторую компоненту)
    print("\nкопим файл диагностики (PR-точки):")
    for frac in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
        k = max(int(frac * len(ys)), 1)
        print(f"  top {frac:.0%}: precision={ys[:k].mean():.3f} recall={ys[:k].sum()/y.sum():.3f}")


if __name__ == "__main__":
    main()
