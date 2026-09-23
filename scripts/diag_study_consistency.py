# -*- coding: utf-8 -*-
"""Диагностика: согласованы ли метки внутри исследования (нужно для агрегации).

Если внутри study метка критерия обычно одинакова на всех снимках, то усреднение
скоров по study — шумоподавление и должно помочь. Если метки смешанные —
усреднение размывает сигнал и навредит.

Также печатается, сколько исследований содержит критерий (для LOSO-оценки).

Запуск:
    cd dxa_qc && python ../scripts/diag_study_consistency.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dxa_qc"))

from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402


def main() -> None:
    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    d = data[real]
    print(f"реальных снимков: {len(d)}, исследований: {d['study_uid'].nunique()}\n")
    print(f"{'критерий':20} {'снимков':>8} {'study':>6} {'pos сним':>9} {'pos study':>10} "
          f"{'смешанных study':>16} {'ср. снимков/study':>18}")
    for name in C.VIOLATIONS:
        key = "spine" if name.startswith("spine") else "femur"
        sub = d[d["region"].str.contains(key)]
        y = sub["viol_" + name].values.astype(float)
        kn = ~np.isnan(y)
        sub = sub[kn]
        y = y[kn].astype(int)
        g = sub["study_uid"].values
        studies = np.unique(g)
        pos_studies = 0
        mixed = 0
        for s in studies:
            v = y[g == s]
            if v.sum() > 0:
                pos_studies += 1
            if 0 < v.sum() < len(v):
                mixed += 1
        print(f"{name:20} {len(y):8d} {len(studies):6d} {int(y.sum()):9d} "
              f"{pos_studies:10d} {mixed:16d} {len(y)/len(studies):18.2f}")


if __name__ == "__main__":
    main()
