# -*- coding: utf-8 -*-
"""Быстрая проверка preprocess.py на реальном DICOM."""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dxa_qc"))
sys.path.insert(0, os.path.join(HERE, ".."))

from src import preprocess as P  # noqa: E402

root = os.path.join(os.environ.get(
    "DXA_DATASET", os.path.join(HERE, "..", "_dsroot")), "исследования")
files = glob.glob(os.path.join(root, "**", "*"), recursive=True)
files = [f for f in files if os.path.isfile(f) and not f.lower().endswith(".xlsx")]
print("файлов:", len(files))
ok = 0
for f in files[:8]:
    ds, arr = P.read_pixels(f)
    disp = P.to_display_float(ds, arr)
    net = P.preprocess_for_net(arr, size=224, ds=ds)
    print(f"{os.path.basename(f):12} shape={arr.shape} disp={disp.shape} "
          f"net={net.shape} dtype={net.dtype} min={net.min():.3f} max={net.max():.3f}")
    ok += 1
print("успешно:", ok)
