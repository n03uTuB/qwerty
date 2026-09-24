# -*- coding: utf-8 -*-
"""Собрать монтаж синтетических артефактов для визуальной проверки."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dxa_qc.src import config as C  # noqa: E402

man = pd.read_csv("out/manifest_art2.csv")
synth = man[man["synthetic"].fillna(False)]
synth = synth[synth["viol_spine_artifacts"].fillna(0) > 0].head(12)
tiles = []
for _, r in synth.iterrows():
    im = Image.open(r["cache_path"]).convert("L").resize((160, 160))
    tiles.append(np.asarray(im))
if tiles:
    n = len(tiles)
    cols = 4
    rows = (n + cols - 1) // cols
    canvas = np.zeros((rows * 160, cols * 160), dtype=np.uint8)
    for i, t in enumerate(tiles):
        rr, cc = divmod(i, cols)
        canvas[rr * 160:(rr + 1) * 160, cc * 160:(cc + 1) * 160] = t
    Image.fromarray(canvas).save("out/artifacts_montage.png")
    print("saved out/artifacts_montage.png", canvas.shape)
