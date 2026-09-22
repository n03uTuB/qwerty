# -*- coding: utf-8 -*-
"""Подбор порогов классификации по OOF-предсказаниям кросс-валидации.

Сохраняет thresholds.json, который затем использует инференс.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import config as C
from . import dataset as ds
from .train import best_threshold


def main():
    manifest = ds.load_manifest(C.MANIFEST_CSV)
    oof_q = np.load(os.path.join(C.ARTIFACTS_DIR, "oof_quality.npy"))
    oof_v = np.load(os.path.join(C.ARTIFACTS_DIR, "oof_violation.npy"))

    # Качество — «сырая» CNN-шкала. Нарушения — гибридная шкала стекера
    # (CNN + геометрия), если она посчитана; иначе тоже CNN.
    stack_v_path = os.path.join(C.ARTIFACTS_DIR, "stack_oof_violation.npy")
    source = "cnn"
    if os.path.isfile(stack_v_path):
        sv = np.load(stack_v_path)
        if (~np.isnan(sv)).sum() > 0:
            oof_v = sv
            source = "stack"
    print("[calibrate] источник OOF-вероятностей (нарушения):", source)

    # пороги калибруем только по реальным снимкам: на синтетике доля нарушений искусственная
    real = ds.real_mask(manifest)
    if not real.all():
        print(f"[calibrate] синтетических строк исключено: {int((~real).sum())}")

    thresholds = {}

    # --- качество (общий порог) ---
    y = manifest["quality"].values.astype(float)
    valid = ~np.isnan(y) & ~np.isnan(oof_q) & real
    if valid.sum() and len(np.unique(y[valid])) > 1:
        thr, f1 = best_threshold(y[valid], oof_q[valid])
        thresholds["quality"] = dict(threshold=float(thr), f1=float(f1),
                                     n=int(valid.sum()))
    else:
        thresholds["quality"] = dict(threshold=0.5, f1=None, n=int(valid.sum()))

    # --- качество по каждой анатомической области ---
    qbr = {}
    for region in C.REGIONS:
        m = (manifest["region"] == region).values & real
        yr = manifest["quality"].values[m].astype(float)
        pr = oof_q[m]
        vr = ~np.isnan(yr) & ~np.isnan(pr)
        if vr.sum() and len(np.unique(yr[vr])) > 1:
            t, f = best_threshold(yr[vr], pr[vr])
            qbr[region] = dict(threshold=float(t), f1=float(f),
                               n=int(vr.sum()), pos=int((yr[vr] > 0.5).sum()))
        else:
            qbr[region] = dict(threshold=thresholds["quality"]["threshold"],
                               f1=None, n=int(vr.sum()))
    thresholds["quality_by_region"] = qbr

    # --- нарушения ---
    vth = {}
    for i, name in enumerate(C.VIOLATIONS):
        col = "viol_" + name
        if col not in manifest:
            continue
        yv = manifest[col].values.astype(float)
        val = ~np.isnan(yv) & ~np.isnan(oof_v[:, i]) & real
        if val.sum() and len(np.unique(yv[val])) > 1:
            thr, f1 = best_threshold(yv[val], oof_v[val, i])
            vth[name] = dict(threshold=float(thr), f1=float(f1),
                             n=int(val.sum()), pos=int((yv[val] > 0.5).sum()))
        else:
            vth[name] = dict(threshold=0.5, f1=None, n=int(val.sum()))
    thresholds["violations"] = vth

    path = os.path.join(C.ARTIFACTS_DIR, "thresholds.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)
    print("[calibrate] сохранено:", path)
    print(json.dumps(thresholds, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
