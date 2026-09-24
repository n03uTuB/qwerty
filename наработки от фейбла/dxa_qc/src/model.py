# -*- coding: utf-8 -*-
"""Мультизадачная нейросеть контроля качества денситометрии.

Одна сеть предсказывает для каждого изображения:
  * anatomical_region  (3 класса: поясничный отдел / левое / правое бедро);
  * quality_class      (бинарно: качественное / есть нарушение);
  * violation_type     (мультилейбл по 5 критериям).

Соответствует пункту «models» из ТЗ. Поддерживаются бэкбоны из torchvision
(resnet18/resnet34, доступны всегда) и любые модели timm (EfficientNetV2,
ConvNeXt, Swin и т.д.) — timm подключается опционально, с graceful fallback.

Также здесь живут:
  * функции потерь (masked BCE и Focal BCE) для борьбы с дисбалансом классов;
  * безопасный TTA (малые повороты и зум) для повышения устойчивости инференса.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C

# --------------------------------------------------------------------------- #
# Реестр бэкбонов
# --------------------------------------------------------------------------- #
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


class _TorchvisionFeatures(nn.Module):
    """Обёртка torchvision-сети: возвращает вектор признаков (B, feat_dim)."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.body = nn.Sequential(*list(net.children())[:-1])  # до global pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x).flatten(1)


def _adapt_first_conv(net: nn.Module, in_chans: int) -> None:
    """Подстроить первый свёрточный слой под нужное число каналов (1 вместо 3)."""
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


def _build_backbone(name: str, pretrained: bool, in_chans: int) -> Tuple[nn.Module, int]:
    """Собрать бэкбон -> (модуль, выдающий (B, feat_dim), feat_dim)."""
    if name in _TORCHVISION_BACKBONES:
        import torchvision.models as tvm

        if name == "resnet18":
            weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = tvm.resnet18(weights=weights)
        else:
            weights = tvm.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            net = tvm.resnet34(weights=weights)
        feat_dim = net.fc.in_features
        _adapt_first_conv(net, in_chans)
        return _TorchvisionFeatures(net), feat_dim

    # остальные — через timm
    try:
        import timm

        net = timm.create_model(name, pretrained=pretrained, num_classes=0,
                                in_chans=in_chans)
        feat_dim = int(getattr(net, "num_features", 0)) or int(net(torch.zeros(1, in_chans, 64, 64)).shape[1])
        return net, feat_dim
    except Exception as e:  # timm нет или модели нет в реестре
        raise ValueError(
            "Бэкбон %r недоступен (timm не установлен или имя неверно). "
            "Доступные: %s. Исходная ошибка: %s" % (name, list_backbones(), e))


# --------------------------------------------------------------------------- #
# Модель
# --------------------------------------------------------------------------- #
class MultiTaskNet(nn.Module):
    def __init__(self, backbone: str = C.BACKBONE, pretrained: bool = True,
                 in_chans: int = 1, dropout: float = C.DROPOUT):
        super().__init__()
        self.backbone_name = backbone
        self.in_chans = in_chans
        self.backbone, feat_dim = _build_backbone(backbone, pretrained, in_chans)
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
            quality=self.head_quality(feat),           # (B, R)
            violation=self.head_violation(feat),        # (B, V)
        )


def build_model(pretrained: bool = True, backbone: Optional[str] = None,
                in_chans: int = 1, dropout: Optional[float] = None) -> MultiTaskNet:
    return MultiTaskNet(backbone=backbone or C.BACKBONE, pretrained=pretrained,
                        in_chans=in_chans,
                        dropout=C.DROPOUT if dropout is None else dropout)


def load_weights(model: MultiTaskNet, path: str, device: str = "cpu") -> MultiTaskNet:
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model


def infer_backbone_from_weights(path: str, device: str = "cpu") -> Optional[str]:
    """Определить бэкбон torchvision по ключам state_dict (resnet18/resnet34).

    Нужно, чтобы инференс мог грузить чекпоинт, обученный с другим бэкбоном,
    не требуя ручного указания. resnet34 глубже: в layer4 есть блок body.7.2,
    у resnet18 максимум body.7.1; также у resnet34 блоки body.4.2 / body.5.2.
    """
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
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


# --------------------------------------------------------------------------- #
# Функции потерь
# --------------------------------------------------------------------------- #
def masked_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
               pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """BCE с маской валидных меток (NaN-метки исключаются)."""
    loss = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=pos_weight)
    mask = mask.float()
    denom = mask.sum().clamp(min=1.0)
    return (loss * mask).sum() / denom


def focal_bce_with_logits(logits: torch.Tensor, target: torch.Tensor,
                          gamma: float = C.FOCAL_GAMMA, alpha: float = C.FOCAL_ALPHA,
                          pos_weight: Optional[torch.Tensor] = None,
                          mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Focal Loss для бинарной мультилейбл-задачи.

    Понижает вклад «лёгких» примеров и повышает вклад редких нарушений — важно
    при 6–36 положительных примерах на критерий. Совместима с pos_weight и маской.
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
                 pos_w_q: Optional[torch.Tensor], pos_w_v: Optional[torch.Tensor],
                 mode: str = C.QUALITY_LOSS) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Суммарный лосс мультизадачной модели.

    mode = weighted_bce | focal. Возвращает (loss, компоненты).
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
    parts = dict(region=float(l_region), quality=float(l_quality), violation=float(l_viol))
    return loss, parts


# --------------------------------------------------------------------------- #
# TTA (test-time augmentation)
# --------------------------------------------------------------------------- #
def _affine_batch(x: torch.Tensor, angle_deg: float, scale: float) -> torch.Tensor:
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
    """Усреднить предсказания по безопасным аффинным преобразованиям.

    Отражения намеренно не используются: они меняют анатомическую сторону
    (лево/право) бедра и ломают голову региона. Повороты ограничены < 5°,
    чтобы не имитировать реальное нарушение оси позвоночника.
    """
    region = quality = violation = None
    n = 0
    for angle, scale in affines:
        out = model(_affine_batch(x, angle, scale))
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
    """Вернуть (quality (B,R), violation (B,V), region (B,R)) в виде вероятностей."""
    if tta:
        out = forward_tta(model, x)
        return out["quality"], out["violation"], out["region"]
    out = model(x)
    return torch.sigmoid(out["quality"]), torch.sigmoid(out["violation"]), \
        torch.softmax(out["region"], dim=1)


def enable_mc_dropout(model: nn.Module, p: Optional[float] = None) -> nn.Module:
    """Включить dropout в режиме eval для оценки неопределённости (MC-dropout)."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
            if p is not None:
                m.p = p
    return model
