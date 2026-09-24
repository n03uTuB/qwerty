# -*- coding: utf-8 -*-
"""Слой данных гибридного сервиса DXA-QC.

  * :mod:`src.data.manifest`  — построение/чтение манифеста, метки, real_mask;
  * :mod:`src.data.dataset`   — torch Dataset и взвешенный сэмплер;
  * :mod:`src.data.synthetic` — синтетические артефакты v2 (только обучение).
"""
from __future__ import annotations

from .dataset import DXADataset, make_balanced_sampler
from .manifest import build_manifest, load_manifest, real_mask, save_manifest
from .synthetic import add_synthetic, make_synthetic_rows

__all__ = [
    "DXADataset", "make_balanced_sampler",
    "build_manifest", "load_manifest", "real_mask", "save_manifest",
    "add_synthetic", "make_synthetic_rows",
]