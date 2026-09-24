# -*- coding: utf-8 -*-
"""Диагностика метки «укладка»: TP/FP/FN и разбивка по областям (конфигурация v3).

Показывает, что ограничивает F1 объединённой метки: ложные срабатывания или
пропуски, и какой критерий (spine_positioning / femur_positioning) даёт вклад.

Запуск:
    python dxa_qc_work/scripts/diag_ukladka.py
"""
from __future__ import annotations

import os

import numpy as np

import _common  # noqa: F401

from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402

from exp_sweep import CriterionScores  # noqa: E402
from build_v3 import V3_FEATURES, V3_SOURCES  # noqa: E402
from exp_confirm import per_seed_macro  # noqa: E402


def main():
    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    cs = CriterionScores(data, real)
    S = {n: cs.get(n, V3_FEATURES[n], V3_SOURCES[n]) for n in C.VIOLATIONS}

    # пороги v3 из бандла
    import json
    thr = json.load(open(os.path.join(_common.OUT, "v3", "thresholds_v3.json"),
                         encoding="utf-8"))["violations"]
    print("пороги v3:", {k: round(v["threshold"], 3) for k, v in thr.items()})

    idx = np.where(real)[0]
    for name in ("spine_positioning", "femur_positioning"):
        reg = "spine" if name.startswith("spine") else "femur"
        m = idx[data["region"].str.contains(reg).values[idx]]
        y = data["viol_" + name].values.astype(float)[m]
        p = S[name][m]
        t = thr[name]["threshold"]
        pred = (p >= t).astype(int)
        yb = np.nan_to_num(y, nan=0.0).astype(int)
        tp = int(((yb == 1) & (pred == 1)).sum())
        fp = int(((yb == 0) & (pred == 1)).sum())
        fn = int(((yb == 1) & (pred == 0)).sum())
        print(f"\n{name}: n={len(m)} pos={int(yb.sum())} thr={t:.3f}")
        print(f"  TP={tp} FP={fp} FN={fn}")
        print(f"  precision={tp/max(tp+fp,1):.3f} recall={tp/max(tp+fn,1):.3f}")

    # объединённая метка «укладка» (только реальные снимки)
    gt = np.zeros(len(data), dtype=int)
    pr = np.zeros(len(data), dtype=int)
    for name in ("spine_positioning", "femur_positioning"):
        reg = "spine" if name.startswith("spine") else "femur"
        m = data["region"].str.contains(reg).values & real
        yb = np.nan_to_num(data["viol_" + name].values.astype(float), nan=0.0).astype(int)
        t = thr[name]["threshold"]
        p = S[name]
        gt[m] = yb[m]
        pr[m & (p >= t)] = 1
    tp = int(((gt == 1) & (pr == 1)).sum())
    fp = int(((gt == 0) & (pr == 1)).sum())
    fn = int(((gt == 1) & (pr == 0)).sum())
    print(f"\nУКЛАДКА (OR): pos={int(gt[idx].sum())} TP={tp} FP={fp} FN={fn}  "
          f"precision={tp/max(tp+fp,1):.3f} recall={tp/max(tp+fn,1):.3f}  "
          f"F1={2*tp/max(2*tp+fp+fn,1):.3f}")

    # что если пороги сдвинуть (oracle по val) — сколько резерва
    best = 0
    for name in ("spine_positioning", "femur_positioning"):
        reg = "spine" if name.startswith("spine") else "femur"
        m = data["region"].str.contains(reg).values
        p = S[name]
        for t in np.linspace(0.02, 0.9, 89):
            pr2 = np.zeros(len(data), dtype=int)
            for n2 in ("spine_positioning", "femur_positioning"):
                r2 = "spine" if n2.startswith("spine") else "femur"
                mm = data["region"].str.contains(r2).values
                tt = t if n2 == name else thr[n2]["threshold"]
                pr2[mm & (S[n2] >= tt)] = 1
            f = 2 * ((gt == 1) & (pr2 == 1)).sum() / max((gt == 1).sum() + pr2.sum(), 1)
            if f > best:
                best, bt = f, t
    print(f"oracle-порог для {name}: F1={best:.3f} при t={bt:.3f}")


if __name__ == "__main__":
    main()
