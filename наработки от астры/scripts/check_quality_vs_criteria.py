# -*- coding: utf-8 -*-
"""Связь quality_class с критериями в разметке организатора.

Проверяет, является ли quality_class независимой оценкой или ИЛИ критериев.
Это определяет, можно ли считать quality_class = ИЛИ(сработавших критериев).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dxa_qc"))

from src import config as C      # noqa: E402
from src import dataset as ds    # noqa: E402

m = ds.load_manifest(C.MANIFEST_CSV)
m = m[~m.synthetic]
y = m["quality"].values.astype(float)
viol = m[["viol_" + n for n in C.VIOLATIONS]].values.astype(float)
known = ~np.isnan(y)
orv = (np.nansum(viol, axis=1) > 0).astype(float)
match = ((y == orv) & known).sum()
print(f"строк: {int(known.sum())}")
print(f"quality == ИЛИ(нарушений): {int(match)}/{int(known.sum())}")
bad = known & (y != orv)
print(f"  quality=1, но критерии все 0: {int((bad & (y == 1)).sum())}")
print(f"  quality=0, но есть критерий 1: {int((bad & (y == 0)).sum())}")
print()
for n in C.VIOLATIONS:
    print(f"  {n:22} pos={int(np.nansum(m['viol_' + n].values))}")
print(f"  quality pos={int(np.nansum(y))}")
