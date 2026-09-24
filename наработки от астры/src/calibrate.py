# -*- coding: utf-8 -*-
"""Подбор порогов классификации по OOF-предсказаниям кросс-валидации.

Сохраняет artifacts/thresholds.json, который затем использует инференс.
Качество — «сырая» CNN-шкала; нарушения — гибридная шкала стекера, если она
посчитана (stack_oof_violation.npy), иначе CNN. Пороги калибруются только по
реальным снимкам (на синтетике доля нарушений искусственная).
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import config as C
from .data import load_manifest, real_mask
from .metrics import pick_threshold


def main():
    manifest = load_manifest(C.MANIFEST_CSV)
    oof_q = np.load(C.OOF_QUALITY_NPY)
    oof_v = np.load(C.OOF_VIOLATION_NPY)

    source = "cnn"
    if os.path.isfile(C.STACK_OOF_VIOLATION_NPY):
        sv = np.load(C.STACK_OOF_VIOLATION_NPY)
        if (~np.isnan(sv)).sum() > 0:
            oof_v = sv
            source = "stack"
    print("[calibrate] источник OOF (нарушения):", source)
    print("[calibrate] режим порога:", C.THRESHOLD_MODE)

    real = real_mask(manifest)
    if not real.all():
        print(f"[calibrate] синтетических строк исключено: {int((~real).sum())}")

    thresholds = {}

    y = manifest["quality"].values.astype(float)
    valid = ~np.isnan(y) & ~np.isnan(oof_q) & real
    if valid.sum() and len(np.unique(y[valid])) > 1:
        thr, f1 = pick_threshold(y[valid], oof_q[valid])
        thresholds["quality"] = dict(threshold=float(thr), f1=float(f1),
                                     n=int(valid.sum()))
    else:
        thresholds["quality"] = dict(threshold=0.5, f1=None, n=int(valid.sum()))

    qbr = {}
    for region in C.REGIONS:
        m = (manifest["region"] == region).values & real
        yr = manifest["quality"].values[m].astype(float)
        pr = oof_q[m]
        vr = ~np.isnan(yr) & ~np.isnan(pr)
        if vr.sum() and len(np.unique(yr[vr])) > 1:
            t, f = pick_threshold(yr[vr], pr[vr])
            qbr[region] = dict(threshold=float(t), f1=float(f),
                               n=int(vr.sum()), pos=int((yr[vr] > 0.5).sum()))
        else:
            qbr[region] = dict(threshold=thresholds["quality"]["threshold"],
                               f1=None, n=int(vr.sum()))
    thresholds["quality_by_region"] = qbr

    vth = {}
    for i, name in enumerate(C.VIOLATIONS):
        col = "viol_" + name
        if col not in manifest:
            continue
        yv = manifest[col].values.astype(float)
        val = ~np.isnan(yv) & ~np.isnan(oof_v[:, i]) & real
        if val.sum() and len(np.unique(yv[val])) > 1:
            thr, f1 = pick_threshold(yv[val], oof_v[val, i])
            vth[name] = dict(threshold=float(thr), f1=float(f1),
                             n=int(val.sum()), pos=int((yv[val] > 0.5).sum()))
        else:
            vth[name] = dict(threshold=0.5, f1=None, n=int(val.sum()))
    thresholds["violations"] = vth

    with open(C.THRESHOLDS_JSON, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)
    print("[calibrate] сохранено:", C.THRESHOLDS_JSON)
    print(json.dumps(thresholds, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()