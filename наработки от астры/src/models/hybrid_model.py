# -*- coding: utf-8 -*-
"""Гибридная мультизадачная сеть контроля качества DXA.

Одна сеть предсказывает для каждого изображения:
  * ``region``    — 3 класса (поясничный отдел / левое / правое бедро);
  * ``quality``   — регион-зависимый логит качества (R логитов);
  * ``violation`` — мультилейбл по 5 критериям ТЗ.

Это «нейросетевая половина» гибрида: геометрические признаки и стекеры живут
в :mod:`src.stack` и объединяются с этими вероятностями на этапе решения.
Бэкбоны — из :mod:`src.models.backbones` (torchvision всегда, timm опционально).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .. import config as C
from .backbones import build_backbone


class MultiTaskNet(nn.Module):
    """Мультизадачная сеть с тремя головами (region / quality / violation)."""

    def __init__(self, backbone: str = C.BACKBONE, pretrained: bool = True,
                 in_chans: int = 1, dropout: float = C.DROPOUT):
        super().__init__()
        self.backbone_name = backbone
        self.in_chans = in_chans
        self.backbone, feat_dim = build_backbone(backbone, pretrained, in_chans)
        self.feat_dim = feat_dim

        self.dropout = nn.Dropout(dropout)
        self.head_region = nn.Linear(feat_dim, len(C.REGIONS))
        # отдельный логит качества на каждую область (регион-зависимая голова)
        self.head_quality = nn.Linear(feat_dim, len(C.REGIONS))
        self.head_violation = nn.Linear(feat_dim, C.N_VIOLATIONS)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.backbone(x)
        if feat.dim() > 2:
            feat = feat.flatten(1)
        feat = self.dropout(feat)
        return dict(
            region=self.head_region(feat),
            quality=self.head_quality(feat),        # (B, R)
            violation=self.head_violation(feat),    # (B, V)
        )


def build_model(pretrained: bool = True, backbone: Optional[str] = None,
                in_chans: int = 1, dropout: Optional[float] = None) -> MultiTaskNet:
    """Собрать мультизадачную сеть (дефолтный бэкбон — config.BACKBONE)."""
    return MultiTaskNet(backbone=backbone or C.BACKBONE, pretrained=pretrained,
                        in_chans=in_chans,
                        dropout=C.DROPOUT if dropout is None else dropout)


def load_weights(model: MultiTaskNet, path: str, device: str = "cpu") -> MultiTaskNet:
    """Загрузить веса (поддерживает как «голый» state_dict, так и {"model": ...})."""
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model