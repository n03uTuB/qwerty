# -*- coding: utf-8 -*-
"""Dataset и аугментации для обучения CNN-части гибрида.

Изображение читается из PNG-кэша (после modality/VOI LUT — тот же вход, что и
при инференсе), затем проходит :func:`src.preprocess.preprocess_for_net`.
Аугментации «физиологичны»: малые повороты (<4 град), лёгкий масштаб/сдвиг и
яркостный джиттер. Отражения запрещены — они меняют сторону бедра и ломают
голову региона.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

from .. import config as C
from ..preprocess import preprocess_for_net


def _random_affine(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Малый поворот/масштаб/сдвиг через cv2.warpAffine (border replicate)."""
    import cv2

    h, w = img.shape[:2]
    angle = float(rng.uniform(-3.5, 3.5))
    scale = float(rng.uniform(0.95, 1.05))
    tx = float(rng.uniform(-0.03, 0.03) * w)
    ty = float(rng.uniform(-0.03, 0.03) * h)
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    m[0, 2] += tx
    m[1, 2] += ty
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _intensity_jitter(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Лёгкий джиттер яркости/контраста + шум (устойчивость к разным аппаратам)."""
    gain = float(rng.uniform(0.9, 1.1))
    bias = float(rng.uniform(-0.04, 0.04))
    out = img * gain + bias
    if rng.random() < 0.5:
        out = out + rng.normal(0.0, 0.01, size=out.shape).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


class DXADataset(Dataset):
    """Изображения DXA + регион-зависимые метки качества и типов нарушений."""

    def __init__(self, df: pd.DataFrame, train: bool = False,
                 size: Optional[int] = None, image_size: Optional[int] = None,
                 preprocess_mode: Optional[str] = None, seed: int = C.SEED):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.size = int(size or image_size or C.IMAGE_SIZE)
        self.preprocess_mode = preprocess_mode or C.PREPROCESS_MODE
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.df)

    def _load(self, cache_path: str) -> np.ndarray:
        img = np.asarray(Image.open(cache_path).convert("L"), dtype=np.uint8)
        return img.astype(np.float32) / 255.0

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        img = self._load(row["cache_path"])
        if self.train:
            img = _random_affine(img, self.rng)
            img = _intensity_jitter(img, self.rng)
        x = preprocess_for_net(img, mode=self.preprocess_mode, size=self.size)
        x = torch.from_numpy(np.ascontiguousarray(x)).float()

        quality = row["quality"]
        has_q = 0.0 if pd.isna(quality) else 1.0
        q = 0.0 if pd.isna(quality) else float(quality)

        viol = np.array([row.get("viol_" + n, np.nan) for n in C.VIOLATIONS],
                        dtype=np.float32)
        vmask = np.array([row.get("mask_" + n, 0.0) for n in C.VIOLATIONS],
                         dtype=np.float32)
        viol = np.nan_to_num(viol, nan=0.0)

        return dict(
            image=x,
            region=torch.tensor(int(row["region_idx"]), dtype=torch.long),
            quality=torch.tensor(q, dtype=torch.float32),
            quality_mask=torch.tensor(has_q, dtype=torch.float32),
            viol=torch.from_numpy(viol),
            viol_mask=torch.from_numpy(np.nan_to_num(vmask, nan=0.0)),
            index=torch.tensor(int(idx), dtype=torch.long),
        )


def make_balanced_sampler(df: pd.DataFrame,
                          seed: int = C.SEED) -> WeightedRandomSampler:
    """Взвешенный сэмплер: поднимает редкие «положительные» снимки.

    Положительным считается снимок с нарушением качества (``quality == 1``) или
    хотя бы одним сработавшим критерием. Веса — обратные частотам классов.
    """
    pos = np.zeros(len(df), dtype=bool)
    if "quality" in df:
        pos |= (df["quality"].fillna(0).values >= 0.5)
    for n in C.VIOLATIONS:
        col = "viol_" + n
        if col in df:
            pos |= (df[col].fillna(0).values >= 0.5)
    n_pos = max(int(pos.sum()), 1)
    n_neg = max(int((~pos).sum()), 1)
    w = np.where(pos, 1.0 / n_pos, 1.0 / n_neg).astype(np.float64)
    w = torch.as_tensor(w, dtype=torch.double)
    return WeightedRandomSampler(w, num_samples=len(df), replacement=True,
                                 generator=torch.Generator().manual_seed(seed))