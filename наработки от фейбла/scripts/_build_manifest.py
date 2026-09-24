# -*- coding: utf-8 -*-
"""Построить манифест CNN-пайплайна для нашего датасета."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dxa_qc.src import dataset as D  # noqa: E402

ROOT = os.environ.get("DXA_DATASET", os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "_dsroot"))

if __name__ == "__main__":
    df = D.build_manifest(ROOT)
    print(df.head(3).to_string())
    print("synthetic:", int(df.synthetic.sum()) if "synthetic" in df else "n/a")
