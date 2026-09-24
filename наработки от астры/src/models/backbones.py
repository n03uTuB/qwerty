# -*- coding: utf-8 -*-
"""Реестр бэкбонов и автоопределение архитектуры по чекпоинту.

Собрано из двух решений:
  * базовый реестр (``dxa_qc``): ``resnet18``/``resnet34`` через torchvision
    (доступны всегда), остальные — через timm с graceful fallback;
  * ``infer_backbone_from_weights`` (``fable_solution``): читает ``state_dict``
    и определяет архитектуру по глубине блоков ``layer3``/``layer4``. Это
    устраняет рассинхрон «веса обучены с resnet34, а грузятся в resnet18»,
    который иначе даёт молчаливо испорченные предсказания при инференсе.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .. import config as C

# Бэкбоны torchvision: доступны без внешних зависимостей.
_TORCHVISION_BACKBONES = ("resnet18", "resnet34")


def timm_available() -> bool:
    try:
        import timm  # noqa: F401

        return True
    except Exception:
        return False


def list_backbones() -> List[str]:
    """Доступные бэкбоны: torchvision всегда, остальные — при наличии timm."""
    names = list(_TORCHVISION_BACKBONES)
    if timm_available():
        names += [b for b in C.BACKBONE_ALTERNATIVES if b not in names]
    return names


class TorchvisionFeatures(nn.Module):
    """Обёртка torchvision-сети: возвращает вектор признаков (B, feat_dim)."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.body = nn.Sequential(*list(net.children())[:-1])  # до global pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x).flatten(1)


def adapt_first_conv(net: nn.Module, in_chans: int) -> None:
    """Подстроить первый свёрточный слой под нужное число каналов (1 вместо 3).

    Веса ImageNet усредняются по RGB-каналам — так одноканальный вход сохраняет
    предобученные фильтры вместо случайной инициализации.
    """
    if in_chans == 3:
        return
    old = getattr(net, "conv1", None)
    if old is None or old.in_channels == in_chans:
        return
    new = nn.Conv2d(in_chans, old.out_channels, kernel_size=old.kernel_size,
                    stride=old.stride, padding=old.padding, bias=False)
    with torch.no_grad():
        if old.weight.shape[1] >= 3:
            new.weight.copy_(old.weight[:, :3].sum(dim=1, keepdim=True) / 3.0)
        else:
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
    net.conv1 = new


def build_backbone(name: str, pretrained: bool = True,
                   in_chans: int = 1) -> Tuple[nn.Module, int]:
    """Собрать бэкбон -> (модуль, выдающий (B, feat_dim), feat_dim)."""
    if name in _TORCHVISION_BACKBONES:
        import torchvision.models as tvm

        if name == "resnet18":
            weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = tvm.resnet18(weights=weights)
        else:
            weights = tvm.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            net = tvm.resnet34(weights=weights)
        feat_dim = int(net.fc.in_features)
        adapt_first_conv(net, in_chans)
        return TorchvisionFeatures(net), feat_dim

    # остальные — через timm
    try:
        import timm

        net = timm.create_model(name, pretrained=pretrained, num_classes=0,
                                in_chans=in_chans)
        feat_dim = int(getattr(net, "num_features", 0))
        if feat_dim <= 0:
            feat_dim = int(net(torch.zeros(1, in_chans, 64, 64)).shape[1])
        return net, feat_dim
    except Exception as e:  # timm нет или модели нет в реестре
        raise ValueError(
            "Бэкбон %r недоступен (timm не установлен или имя неверно). "
            "Доступные: %s. Исходная ошибка: %s" % (name, list_backbones(), e))


def infer_backbone_from_weights(path: str, device: str = "cpu") -> Optional[str]:
    """Определить бэкбон torchvision по ключам state_dict (resnet18/resnet34).

    resnet34 глубже: в ``layer4`` есть блок ``body.7.2``, у resnet18 максимум
    ``body.7.1``; также у resnet34 присутствуют ``body.4.2`` / ``body.5.2``.
    """
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except Exception:
        return None
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if not isinstance(state, dict):
        return None
    keys = list(state.keys())
    for marker, name in (("backbone.body.7.2.", "resnet34"),
                         ("backbone.body.4.2.", "resnet34"),
                         ("backbone.body.7.1.", "resnet18")):
        if any(k.startswith(marker) for k in keys):
            return name
    return None