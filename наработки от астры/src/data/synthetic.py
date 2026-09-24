# -*- coding: utf-8 -*-
"""Синтетические «посторонние предметы, артефакты и наложения» v2.

Критерий ``spine_artifacts`` редок (17 из 99), поэтому генератор накладывает на
чистые позвоночные снимки правдоподобные нарушения и помечает их
``spine_artifacts=1``. Подтипы (порт идей fable_solution):

  * ``edge_object`` — резкий яркий объект (прямоугольник/диск/линия) у контура
    тела ВНЕ костной маски — металл (молнии, пуговицы, монеты, проволока);
  * ``overlay_band`` — сдвиг горизонтальной полосы (наложение/ghosting);
  * ``dark_spot`` — низкоплотное пятно темнее мягких тканей внутри тела.

ВАЖНО: строки помечаются ``synthetic=True`` и используются ТОЛЬКО в обучении;
метрики/пороги/отчёт считаются по ``real_mask`` (иначе оценка «протекает»).
"""
from __future__ import annotations

import hashlib
import os
from typing import List, Optional

import numpy as np
import pandas as pd

from .. import config as C

_SYNTH_TYPES = ("edge_object", "overlay_band", "dark_spot")
_WEIGHTS = (0.5, 0.35, 0.15)


def _as_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def _bone_mask(values01: np.ndarray, level: float = 0.25) -> np.ndarray:
    from scipy import ndimage

    soft, dense = np.percentile(values01, (50, 98))
    if dense <= soft:
        return np.zeros_like(values01, dtype=bool)
    mask = values01 > soft + (dense - soft) * level
    mask = ndimage.binary_closing(mask, np.ones((5, 5)))
    mask = ndimage.binary_fill_holes(mask)
    return ndimage.binary_opening(mask, np.ones((3, 3)))


def edge_object(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Резкий яркий объект у контура тела ВНЕ кости (металл)."""
    from scipy import ndimage

    out = _as_uint8(img).astype(np.float32)
    values = out / 255.0
    body = values > 0.12
    rows, cols = out.shape
    near_edge = (ndimage.binary_dilation(body, np.ones((15, 15)))
                 & ~ndimage.binary_erosion(body, np.ones((15, 15))))
    outside_bone = ~ndimage.binary_dilation(_bone_mask(values, 0.25), np.ones((9, 9)))
    cand = np.argwhere(near_edge & outside_bone)
    if len(cand) >= 30:
        cy, cx = cand[rng.integers(len(cand))]
    else:
        side = int(rng.integers(0, 4))
        if side == 0:
            cy, cx = rng.uniform(0, rows * 0.12), rng.uniform(0, cols)
        elif side == 1:
            cy, cx = rng.uniform(rows * 0.88, rows), rng.uniform(0, cols)
        elif side == 2:
            cy, cx = rng.uniform(0, rows), rng.uniform(0, cols * 0.12)
        else:
            cy, cx = rng.uniform(0, rows), rng.uniform(cols * 0.88, cols)

    target = float(np.percentile(out, rng.uniform(96.0, 99.7)))
    yy, xx = np.mgrid[0:rows, 0:cols]
    dy, dx = yy - cy, xx - cx
    ang = rng.uniform(0, np.pi)
    ry = dy * np.cos(ang) + dx * np.sin(ang)
    rx = -dy * np.sin(ang) + dx * np.cos(ang)
    shape = int(rng.integers(0, 3))
    if shape == 0:          # прямоугольник (молния, пуговица)
        h, w = rng.uniform(3, 10), rng.uniform(3, 14)
        m = (np.abs(ry) <= h / 2) & (np.abs(rx) <= w / 2)
    elif shape == 1:        # диск (монета, клипса)
        r = rng.uniform(3, 7)
        m = (dy ** 2 + dx ** 2) <= r * r
    else:                   # тонкая линия (проволока, пирсинг)
        h, w = rng.uniform(0.8, 2.2), rng.uniform(14, 45)
        m = (np.abs(ry) <= h) & (np.abs(rx) <= w / 2)
    patch = ndimage.gaussian_filter(m.astype(np.float32), sigma=0.4)  # резкие края
    out = out * (1 - patch) + target * patch
    return out.clip(0, 255).astype(np.uint8)


def overlay_band(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Наложение/дублирование: сдвиг горизонтальной полосы (ghosting, движение)."""
    out = _as_uint8(img).astype(np.float32)
    rows, cols = out.shape
    h = max(3, int(rng.uniform(0.06, 0.20) * rows))
    y0 = int(rng.uniform(0, max(1, rows - h)))
    shift = int(rng.choice([-1, 1]) * rng.uniform(3, 12))
    alpha = rng.uniform(0.25, 0.6)
    band = out[y0:y0 + h].copy()
    out[y0:y0 + h] = (1 - alpha) * band + alpha * np.roll(band, shift, axis=1)
    return out.clip(0, 255).astype(np.uint8)


def dark_spot(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Низкоплотное пятно темнее мягких тканей внутри тела (газ/содержимое)."""
    from scipy import ndimage

    out = _as_uint8(img).astype(np.float32)
    values = out / 255.0
    body = values > 0.12
    cand = np.argwhere(body)
    rows, cols = out.shape
    if len(cand) < 30:
        return out.clip(0, 255).astype(np.uint8)
    cy, cx = cand[rng.integers(len(cand))]
    yy, xx = np.mgrid[0:rows, 0:cols]
    r = rng.uniform(5, 14)
    blob = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r
    patch = ndimage.gaussian_filter(blob.astype(np.float32), sigma=2.0)
    target = float(np.percentile(out, rng.uniform(5, 20)))
    out = out * (1 - patch) + target * patch
    return out.clip(0, 255).astype(np.uint8)


_APPLIERS = {"edge_object": edge_object, "overlay_band": overlay_band,
             "dark_spot": dark_spot}


def make_synthetic_rows(df: pd.DataFrame, out_dir: str, per_source: int = 2,
                        rng: Optional[np.random.Generator] = None,
                        max_sources: Optional[int] = None) -> List[dict]:
    """Сгенерировать синтетические строки-артефакты из реальных снимков позвоночника."""
    from PIL import Image

    if rng is None:
        rng = np.random.default_rng(C.SEED)
    os.makedirs(out_dir, exist_ok=True)

    src = df[(df["region"] == C.REGION_SPINE) & (~df.get("synthetic", False))]
    src = src[src["cache_path"].map(os.path.isfile)]
    if max_sources is not None and len(src) > max_sources:
        src = src.sample(n=max_sources, random_state=C.SEED)

    idx_art = C.VIOLATION_IDX["spine_artifacts"]
    probs = np.asarray(_WEIGHTS, dtype=float)
    probs = probs / probs.sum()
    rows: List[dict] = []
    for _, row in src.iterrows():
        base = _as_uint8(np.asarray(Image.open(row["cache_path"]).convert("L")))
        for k in range(per_source):
            kind = str(rng.choice(_SYNTH_TYPES, p=probs))
            aug = _APPLIERS[kind](base, rng)
            name = hashlib.md5(
                (str(row["image_uid"]) + f"|{kind}|{k}").encode("utf-8")
            ).hexdigest() + ".png"
            path = os.path.join(out_dir, name)
            Image.fromarray(aug).save(path)

            viol = np.full(C.N_VIOLATIONS, np.nan, dtype=np.float32)
            mask = np.zeros(C.N_VIOLATIONS, dtype=np.float32)
            viol[idx_art] = 1.0
            mask[idx_art] = 1.0
            new = dict(row)
            new.update(image_uid=f"synth_{name[:-4]}", source_path=path,
                       cache_path=path, quality=1.0, synthetic=True)
            for i, nm in enumerate(C.VIOLATIONS):
                new["viol_" + nm] = viol[i]
                new["mask_" + nm] = mask[i]
            rows.append(new)
    return rows


def add_synthetic(df: pd.DataFrame, out_dir: Optional[str] = None,
                  per_source: int = 2, seed: int = C.SEED,
                  max_sources: Optional[int] = None) -> pd.DataFrame:
    """Вернуть манифест с добавленными синтетическими строками."""
    out_dir = out_dir or os.path.join(C.CACHE_DIR, "synthetic")
    rng = np.random.default_rng(seed)
    rows = make_synthetic_rows(df, out_dir, per_source=per_source, rng=rng,
                               max_sources=max_sources)
    if not rows:
        return df
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)