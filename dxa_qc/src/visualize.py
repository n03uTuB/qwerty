# -*- coding: utf-8 -*-
"""Визуализация объяснений модели (дополнительный функционал ТЗ).

Реализован Grad-CAM для головы качества/нарушений: тепловая карта поверх
исходного изображения, а также контур области интереса.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from . import dicom_io
from . import model as model_lib


class GradCAM:
    """Grad-CAM для последнего свёрточного блока ResNet."""

    def __init__(self, model: model_lib.MultiTaskNet):
        self.model = model
        self.activations = None
        self.gradients = None
        self._register()

    def _register(self):
        target = None
        for m in self.model.backbone.modules():
            if isinstance(m, torch.nn.Conv2d):
                target = m
        self._target = target
        target.register_forward_hook(self._fwd)
        target.register_full_backward_hook(self._bwd)

    def _fwd(self, module, inp, out):
        self.activations = out.detach()

    def _bwd(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, head: str = "quality",
                 class_idx: Optional[int] = None) -> np.ndarray:
        self.model.zero_grad()
        out = self.model(x)
        if head == "quality":
            ridx = class_idx if class_idx is not None else 0
            score = out["quality"][0, ridx]
        elif head == "violation":
            idx = class_idx if class_idx is not None else 0
            score = out["violation"][0, idx]
        else:
            idx = class_idx if class_idx is not None else 0
            score = out["region"][0, idx]
        score.backward()

        act = self.activations[0]
        grad = self.gradients[0]
        weights = grad.mean(dim=(1, 2), keepdim=True)
        cam = (weights * act).sum(0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu().numpy()


def overlay_cam(arr: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Наложить тепловую карту на нормализованное изображение (RGB uint8)."""
    import cv2

    img = dicom_io.normalize_uint8(arr)
    h, w = img.shape
    cam_r = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap((cam_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    base = np.stack([img] * 3, axis=-1)
    out = (base * (1 - alpha) + heat * alpha).astype(np.uint8)
    return out


def save_overlay(controller, path: str, out_png: str,
                 head: str = "quality", class_idx: Optional[int] = None) -> str:
    """Сохранить тепловую карту для изображения (PIL — кириллица-safe)."""
    from PIL import Image

    _ds, arr = dicom_io.read_dicom(path)
    arr = np.asarray(arr)
    model = controller.models[0]
    cam = GradCAM(model)
    img = dicom_io.preprocess(arr, controller.size)
    x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(controller.device)
    c = cam(x, head=head, class_idx=class_idx)
    rgb = overlay_cam(arr, c)
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    Image.fromarray(rgb).save(out_png)
    return out_png
