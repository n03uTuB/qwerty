# -*- coding: utf-8 -*-
"""Мультизадачная нейросеть контроля качества денситометрии.

Одна сеть предсказывает для каждого изображения:
  * anatomical_region  (3 класса: поясничный отдел / левое / правое бедро);
  * quality_class      (бинарно: качественное / есть нарушение);
  * violation_type     (мультилейбл по 5 критериям).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import config as C


class MultiTaskNet(nn.Module):
    def __init__(self, backbone: str = C.BACKBONE, pretrained: bool = True,
                 in_chans: int = 1, dropout: float = 0.3):
        super().__init__()
        import torchvision.models as tvm

        if backbone == "resnet18":
            weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = tvm.resnet18(weights=weights)
            feat_dim = net.fc.in_features
        elif backbone == "resnet34":
            weights = tvm.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            net = tvm.resnet34(weights=weights)
            feat_dim = net.fc.in_features
        else:
            raise ValueError("unknown backbone: %s" % backbone)

        # адаптируем первый свёрточный слой под 1 канал
        if in_chans != 3:
            old = net.conv1
            new = nn.Conv2d(in_chans, old.out_channels, kernel_size=old.kernel_size,
                            stride=old.stride, padding=old.padding, bias=False)
            with torch.no_grad():
                new.weight.copy_(old.weight.sum(dim=1, keepdim=True) / 3.0)
            net.conv1 = new

        self.backbone = nn.Sequential(*list(net.children())[:-1])  # до global pool
        self.dropout = nn.Dropout(dropout)
        self.head_region = nn.Linear(feat_dim, len(C.REGIONS))
        # отдельный логит качества на каждую область (регион-зависимая голова)
        self.head_quality = nn.Linear(feat_dim, len(C.REGIONS))
        self.head_violation = nn.Linear(feat_dim, C.N_VIOLATIONS)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x).flatten(1)
        feat = self.dropout(feat)
        return dict(
            region=self.head_region(feat),
            quality=self.head_quality(feat),           # (B, R)
            violation=self.head_violation(feat),        # (B, V)
        )


def build_model(pretrained: bool = True) -> MultiTaskNet:
    return MultiTaskNet(pretrained=pretrained)


def load_weights(model: MultiTaskNet, path: str, device: str = "cpu") -> MultiTaskNet:
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model
