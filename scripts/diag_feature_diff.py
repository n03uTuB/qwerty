# -*- coding: utf-8 -*-
"""Диагностика расхождений признаков dxa_qc vs dxa_real: по каждому признаку —
максимум и медиана относительной ошибки, плюс пример худшего снимка."""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# корень набора данных: переменная окружения DXA_DATASET, иначе ../_dsroot
DATASET = os.environ.get("DXA_DATASET", os.path.join(HERE, "..", "_dsroot"))
sys.path.insert(0, os.path.join(HERE, "..", "dxa_qc"))
sys.path.insert(0, os.path.join(HERE, ".."))

from src import config as C          # noqa: E402
from src import dataset as ds        # noqa: E402
from src import dicom_io             # noqa: E402
from dxa_real import features as RF  # noqa: E402
from dxa_real.data import load_dataset  # noqa: E402

RENAME = {"spine_midline_angle": "spine_midline_deg"}
KEYS = ["spine_midline_angle", "spine_iliac_signal", "spine_ribs_signal",
        "spine_vertebra_peaks", "femur_trochanter_bulge", "femur_troch_area_mm2",
        "femur_neck_width_mm", "femur_height_cm", "femur_bone_ratio",
        "femur_margin_min_cm"]

manifest = ds.load_manifest(C.MANIFEST_CSV)
feat = pd.read_csv(os.path.join(C.ARTIFACTS_DIR, "features.csv"))
keep = ["image_uid"] + [c for c in feat.columns if c not in ("study_uid", "region", "image_uid")]
merged = manifest.merge(feat[keep], on="image_uid", how="inner")

by_hash = {}
for _, r in merged.iterrows():
    try:
        _ds, arr = dicom_io.read_dicom(r["source_path"])
    except Exception:
        continue
    by_hash[hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()] = (r, arr)

per_key = {k: [] for k in KEYS}
worst_example = {}
for im in load_dataset(DATASET):
    h = hashlib.md5(np.ascontiguousarray(im.array).tobytes()).hexdigest()
    got = by_hash.get(h)
    if got is None:
        continue
    r, arr = got
    ref = RF.compute(im)
    for k in KEYS:
        rk = RENAME.get(k, k)
        if k not in r or rk not in ref:
            continue
        a, b = float(r[k]), float(ref[rk])
        d = abs(a - b) / max(abs(b), 1.0)
        per_key[k].append(d)
        if k not in worst_example or d > worst_example[k][0]:
            worst_example[k] = (d, a, b, r["image_uid"][:20], im.array.dtype,
                                im.array.max(), arr.dtype, arr.max(),
                                tuple(np.round(im.spacing, 4)), (r["spacing_x"], r["spacing_y"]))

print(f"{'признак':26} {'макс':>10} {'медиана':>10} {'>1%':>6}")
for k in KEYS:
    v = np.array(per_key[k])
    print(f"{k:26} {v.max():10.2e} {np.median(v):10.2e} {int((v > 0.01).sum()):6d}")

# сколько изображений вообще расходятся (ожидаем ~число спорных дублей)
n_bad = sum(1 for i in range(len(per_key[KEYS[0]]))
            if any(per_key[k][i] > 1e-3 for k in KEYS))
print(f"\nизображений с расхождением > 0.1%: {n_bad} из {len(per_key[KEYS[0]])}")

print("\nхудшие примеры:")
for k in KEYS:
    if k in worst_example and worst_example[k][0] > 1e-2:
        d, a, b, uid, adt, amax, qdt, qmax, sp_real, sp_qc = worst_example[k]
        print(f"  {k}: rel={d:.2e} qc={a:.4g} real={b:.4g}  uid={uid}")
        print(f"      real: dtype={adt} max={amax} spacing={sp_real}")
        print(f"      qc:   dtype={qdt} max={qmax} spacing={sp_qc}")
