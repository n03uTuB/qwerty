# -*- coding: utf-8 -*-
"""Совместимость: исторический плоский модуль ``src.dataset``.

После рефакторинга пакет разнесён на :mod:`src.data.manifest` (манифест, метки,
``real_mask``) и :mod:`src.data.dataset` (torch Dataset). Этот модуль сохраняет
старый импорт ``from src import dataset as ds`` для скриптов прошлых итераций.
"""
from __future__ import annotations

from .data import (DXADataset, add_synthetic, build_manifest, load_manifest,
                   make_balanced_sampler, make_synthetic_rows, real_mask,
                   save_manifest)

__all__ = [
    "DXADataset", "make_balanced_sampler",
    "build_manifest", "load_manifest", "real_mask", "save_manifest",
    "add_synthetic", "make_synthetic_rows",
]