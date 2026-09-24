# -*- coding: utf-8 -*-
"""Функции потерь для мультизадачной сети DXA-QC.

  * :func:`masked_bce` — BCE с маской валидных меток (NaN-метки исключаются);
  * :func:`focal_bce_with_logits` — Focal Loss для сильного дисбаланса
    (6–36 положительных примеров на критерий);
  * :func:`compute_loss` — суммарный лосс (region + quality + violation).

Совместимы с ``pos_weight`` (веса классов по фолду) и с маской областей.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .. import config as C


def masked_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
               pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """BCE с маской валидных меток (NaN-метки исключаются)."""
    loss = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=pos_weight)
    mask = mask.float()
    denom = mask.sum().clamp(min=1.0)
    return (loss * mask).sum() / denom


def focal_bce_with_logits(logits: torch.Tensor, target: torch.Tensor,
                          gamma: float = C.FOCAL_GAMMA,
                          alpha: float = C.FOCAL_ALPHA,
                          pos_weight: Optional[torch.Tensor] = None,
                          mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Focal Loss для бинарной мультилейбл-задачи.

    Понижает вклад «лёгких» примеров и повышает вклад редких нарушений.
    Совместима с ``pos_weight`` и маской.
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none",
                                             pos_weight=pos_weight)
    p = torch.sigmoid(logits)
    pt = p * target + (1.0 - p) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    loss = alpha_t * (1.0 - pt).clamp(min=1e-6).pow(gamma) * bce
    if mask is not None:
        mask = mask.float()
        return (loss * mask).sum() / mask.sum().clamp(min=1.0)
    return loss.mean()


def compute_loss(out: Dict[str, torch.Tensor], region: torch.Tensor,
                 quality: torch.Tensor, quality_mask: torch.Tensor,
                 viol: torch.Tensor, viol_mask: torch.Tensor,
                 pos_w_q: Optional[torch.Tensor],
                 pos_w_v: Optional[torch.Tensor],
                 mode: str = C.QUALITY_LOSS
                 ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Суммарный лосс мультизадачной модели.

    ``mode`` = ``weighted_bce`` | ``focal``. Голова качества — регион-зависимая:
    берётся логит своей анатомической области, поэтому и ``pos_weight``
    выбирается по области.
    """
    l_region = F.cross_entropy(out["region"], region)
    q_logit = out["quality"].gather(1, region.unsqueeze(1)).squeeze(1)
    pw_q = pos_w_q.to(q_logit.device)[region] if pos_w_q is not None else None
    pw_v = pos_w_v.to(viol.device) if pos_w_v is not None else None

    if mode == "focal":
        l_quality = focal_bce_with_logits(q_logit, quality, pos_weight=pw_q,
                                          mask=quality_mask)
        l_viol = focal_bce_with_logits(out["violation"], viol, pos_weight=pw_v,
                                       mask=viol_mask)
    else:
        l_quality = masked_bce(q_logit, quality, quality_mask, pw_q)
        l_viol = masked_bce(out["violation"], viol, viol_mask, pw_v)

    loss = l_region + l_quality + l_viol
    parts = dict(region=float(l_region), quality=float(l_quality),
                 violation=float(l_viol))
    return loss, parts