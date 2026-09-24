# -*- coding: utf-8 -*-
"""Безопасный TTA (test-time augmentation) для инференса и оценки.

Отражения НЕ используются: они меняют анатомическую сторону (лево/право) бедра
и ломают голову региона. Повороты ограничены < 5°, чтобы не имитировать реальное
нарушение оси позвоночника (допуск ТЗ — 5°).
"""
from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config as C


def affine_batch(x: torch.Tensor, angle_deg: float, scale: float) -> torch.Tensor:
    """Батчевый поворот+зум через affine_grid/grid_sample (без внешних зависимостей)."""
    if angle_deg == 0.0 and scale == 1.0:
        return x
    b = x.shape[0]
    theta = torch.zeros(b, 2, 3, device=x.device, dtype=x.dtype)
    a = math.radians(angle_deg)
    cos, sin = math.cos(a) * scale, math.sin(a) * scale
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="border")


@torch.no_grad()
def forward_tta(model: nn.Module, x: torch.Tensor,
                affines: Sequence[Tuple[float, float]] = tuple(C.TTA_AFFINES)
                ) -> Dict[str, torch.Tensor]:
    """Усреднить предсказания по безопасным аффинным преобразованиям."""
    region = quality = violation = None
    n = 0
    for angle, scale in affines:
        out = model(affine_batch(x, angle, scale))
        r = torch.softmax(out["region"], dim=1)
        q = torch.sigmoid(out["quality"])
        v = torch.sigmoid(out["violation"])
        region = r if region is None else region + r
        quality = q if quality is None else quality + q
        violation = v if violation is None else violation + v
        n += 1
    return dict(region=region / n, quality=quality / n, violation=violation / n)


@torch.no_grad()
def predict_probs(model: nn.Module, x: torch.Tensor, tta: bool = False
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Вернуть (quality (B,R), violation (B,V), region (B,R)) как вероятности."""
    if tta:
        out = forward_tta(model, x)
        return out["quality"], out["violation"], out["region"]
    out = model(x)
    return (torch.sigmoid(out["quality"]), torch.sigmoid(out["violation"]),
            torch.softmax(out["region"], dim=1))


def enable_mc_dropout(model: nn.Module, p=None) -> nn.Module:
    """Включить dropout в режиме eval для оценки неопределённости (MC-dropout)."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
            if p is not None:
                m.p = p
    return model