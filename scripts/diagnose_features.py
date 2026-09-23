# -*- coding: utf-8 -*-
"""Диагностика: AUC каждого отдельного признака по каждому критерию ТЗ.

Быстро (без кросс-валидации) показывает, где модель на признаках слабее
лучшего одиночного признака, и какие признаки вообще полезны.

    python scripts/diagnose_features.py --root "...\\_dsroot"
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import roc_auc_score                      # noqa: E402
from dxa_real import features as F                             # noqa: E402
from dxa_real.data import LABELS, load_dataset                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--top", type=int, default=6)
    args = ap.parse_args()

    images = load_dataset(args.root)
    table = F_import_build(images)

    for region in ("spine", "femur"):
        keys = F.SPINE_KEYS if region == "spine" else F.FEMUR_KEYS
        keys = keys + F.COMMON_KEYS
        sub = table[table["region"] == region]
        print(f"\n{'='*90}\nОБЛАСТЬ: {region}  (n={len(sub)})\n{'='*90}")
        for label in LABELS:
            if label not in sub:
                continue
            y = sub[label].values.astype(float)
            known = ~np.isnan(y)
            if known.sum() == 0 or len(np.unique(y[known])) < 2:
                continue
            scored = []
            for k in keys:
                if k not in sub:
                    continue
                x = sub[k].values.astype(float)
                ok = known & ~np.isnan(x)
                if ok.sum() < 10 or len(np.unique(x[ok])) < 2:
                    continue
                auc = roc_auc_score(y[ok], x[ok])
                scored.append((abs(auc - 0.5) + 0.5, auc, k))
            scored.sort(reverse=True)
            print(f"\n  Критерий: {label}  (pos={int(y[known].sum())}/{int(known.sum())})")
            for _s, auc, k in scored[:args.top]:
                print(f"      {k:28} AUC={auc:.3f}")


def F_import_build(images):
    from dxa_real.evaluate import build_table
    return build_table(images)


if __name__ == "__main__":
    main()
