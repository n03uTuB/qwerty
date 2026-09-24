# -*- coding: utf-8 -*-
"""Модельный слой гибридного сервиса DXA-QC.

Экспортирует:
  * :mod:`src.models.backbones`    — реестр бэкбонов и автоопределение по весам;
  * :mod:`src.models.hybrid_model` — мультизадачная сеть (3 головы);
  * :mod:`src.models.losses`       — masked BCE и Focal Loss;
  * :mod:`src.models.tta`          — безопасный TTA (малые повороты/зум).
"""
from __future__ import annotations

from .backbones import (adapt_first_conv, build_backbone,
                        infer_backbone_from_weights, list_backbones,
                        timm_available)
from .hybrid_model import MultiTaskNet, build_model, load_weights
from .losses import compute_loss, focal_bce_with_logits, masked_bce
from .tta import forward_tta, predict_probs

__all__ = [
    "adapt_first_conv", "build_backbone", "infer_backbone_from_weights",
    "list_backbones", "timm_available",
    "MultiTaskNet", "build_model", "load_weights",
    "compute_loss", "focal_bce_with_logits", "masked_bce",
    "forward_tta", "predict_probs",
]