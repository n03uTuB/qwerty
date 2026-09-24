# -*- coding: utf-8 -*-
"""Синтетические «посторонние предметы, артефакты и наложения» v2.

Почему v2. Прошлая попытка (paste_metal) ВРЕДИЛА: яркий размытый шар в мягких
тканях модель выучивала, а на реальных снимках не срабатывала. Причины провала:
центр кадра вместо периферии, максимум яркости вместо реалистичной, один блоб
вместо разнообразия форм.

Реальные DXA-нарушения этой метки (по разметке: «наличие посторонних предметов,
выраженных артефактов или наложений»):
  * металл (молнии, пуговицы, монеты, проволока) — с РЕЗКИМИ краями, у контура
    тела или края кадра, часто несколько объектов;
  * наложения/движение — дублирование полосы, «ступенька» на контуре (ghosting);
  * низкоплотные зоны (газ/содержимое) — пятна темнее мягких тканей.

Здесь реализованы три подтипа, смешанные случайно:
  * edge_object — резкий яркий объект (прямоугольник/диск/линия) у контура тела,
    яркость берётся из верхних процентилей кадра, а не максимум;
  * overlay_band — сдвиг горизонтальной полосы (дублирование/наложение);
  * dark_spot — низкоплотное пятно в теле.

Гарантии честности те же, что и в synth_real: study_uid наследуется (тот же фолд),
метка ставится только соответствующему критерию, снимки помечаются synthetic.
"""
from __future__ import annotations

import copy

import numpy as np
from scipy import ndimage

from .data import LABEL_FOREIGN, Image
from .features import bone_mask, normalize


def _derive(image: Image, array: np.ndarray, suffix: str, labels: dict) -> Image:
    clone = copy.copy(image)
    clone.array = array
    clone.image_uid = f"{image.image_uid}:{suffix}"
    clone.labels = dict(labels)
    clone.quality = int(any(labels.values()))
    clone.synthetic = True
    clone.source_uid = image.image_uid
    return clone


def edge_object(image: Image, rng: np.random.Generator) -> np.ndarray:
    """Резкий яркий объект у контура тела / края кадра (металл)."""
    array = image.array.astype(np.float32)
    values = normalize(array)
    body = values > 0.12
    rows, cols = array.shape

    near_edge = (ndimage.binary_dilation(body, np.ones((15, 15)))
                 & ~ndimage.binary_erosion(body, np.ones((15, 15))))
    outside_bone = ~ndimage.binary_dilation(bone_mask(values, 0.25), np.ones((9, 9)))
    candidates = np.argwhere(near_edge & outside_bone)
    if len(candidates) >= 30:
        cy, cx = candidates[rng.integers(len(candidates))]
    else:
        # запасной вариант: у случайного края кадра
        side = rng.integers(0, 4)
        if side == 0:
            cy, cx = rng.uniform(0, rows * 0.12), rng.uniform(0, cols)
        elif side == 1:
            cy, cx = rng.uniform(rows * 0.88, rows), rng.uniform(0, cols)
        elif side == 2:
            cy, cx = rng.uniform(0, rows), rng.uniform(0, cols * 0.12)
        else:
            cy, cx = rng.uniform(0, rows), rng.uniform(cols * 0.88, cols)

    # яркость: верхние процентили кадра, а не максимум (кость уже упирается в максимум)
    target = float(np.percentile(array, rng.uniform(96.0, 99.7)))

    yy, xx = np.mgrid[0:rows, 0:cols]
    dy, dx = yy - cy, xx - cx
    angle = rng.uniform(0, np.pi)
    ry = dy * np.cos(angle) + dx * np.sin(angle)
    rx = -dy * np.sin(angle) + dx * np.cos(angle)

    shape = int(rng.integers(0, 3))
    if shape == 0:      # прямоугольник (молния, пуговица)
        h, w = rng.uniform(3, 10), rng.uniform(3, 14)
        mask = (np.abs(ry) <= h / 2) & (np.abs(rx) <= w / 2)
    elif shape == 1:    # диск (монета, клипса)
        r = rng.uniform(3, 7)
        mask = (dy ** 2 + dx ** 2) <= r * r
    else:               # тонкая линия (проволока, пирсинг)
        h, w = rng.uniform(0.8, 2.2), rng.uniform(14, 45)
        mask = (np.abs(ry) <= h) & (np.abs(rx) <= w / 2)

    patch = ndimage.gaussian_filter(mask.astype(np.float32), sigma=0.4)  # резкие края
    array = array * (1 - patch) + target * patch
    return array.clip(0, 255).astype(image.array.dtype)


def overlay_band(image: Image, rng: np.random.Generator) -> np.ndarray:
    """Наложение/дублирование: сдвиг горизонтальной полосы (ghosting, движение)."""
    array = image.array.astype(np.float32)
    rows, cols = array.shape
    h = int(rng.uniform(0.06, 0.20) * rows)
    if h < 3:
        h = 3
    y0 = int(rng.uniform(0, max(1, rows - h)))
    shift = int(rng.choice([-1, 1]) * rng.uniform(3, 12))
    alpha = rng.uniform(0.25, 0.6)
    band = array[y0:y0 + h].copy()
    shifted = np.roll(band, shift, axis=1)
    array[y0:y0 + h] = (1 - alpha) * band + alpha * shifted
    return array.clip(0, 255).astype(image.array.dtype)


def dark_spot(image: Image, rng: np.random.Generator) -> np.ndarray:
    """Низкоплотная зона в теле (газ/содержимое) — пятно темнее мягких тканей."""
    array = image.array.astype(np.float32)
    values = normalize(array)
    body = values > 0.12
    candidates = np.argwhere(body)
    rows, cols = array.shape
    if len(candidates) < 30:
        return array.clip(0, 255).astype(image.array.dtype)
    cy, cx = candidates[rng.integers(len(candidates))]
    yy, xx = np.mgrid[0:rows, 0:cols]
    r = rng.uniform(5, 14)
    blob = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r
    patch = ndimage.gaussian_filter(blob.astype(np.float32), sigma=2.0)
    target = float(np.percentile(array, rng.uniform(5, 20)))
    array = array * (1 - patch) + target * patch
    return array.clip(0, 255).astype(image.array.dtype)


SUBTYPES = ("edge_object", "overlay_band", "dark_spot")


def build_artifacts(images: list[Image], rng: np.random.Generator, per_type: int = 40,
                    weights=(0.5, 0.35, 0.15)) -> list[Image]:
    """Синтетические «предметы/артефакты/наложения» из чистых снимков позвоночника."""
    spine = [im for im in images if im.region == "spine"
             and bool(im.labels) and im.quality == 0 and not any(im.labels.values())]
    generated: list[Image] = []
    if not spine:
        return generated
    fn = {"edge_object": edge_object, "overlay_band": overlay_band, "dark_spot": dark_spot}
    probs = np.asarray(weights, dtype=float)
    probs = probs / probs.sum()
    for _ in range(per_type):
        subtype = str(rng.choice(SUBTYPES, p=probs))
        source = spine[int(rng.integers(len(spine)))]
        array = fn[subtype](source, rng)
        labels = {**{k: 0 for k in source.labels}, LABEL_FOREIGN: 1}
        generated.append(_derive(source, array, f"art_{subtype}", labels))
    return generated


def summary(generated: list[Image]) -> str:
    from collections import Counter

    counts = Counter(im.image_uid.split(":")[-1] for im in generated)
    lines = [f"синтетических артефактов: {len(generated)}"]
    lines += [f"  {name}: {count}" for name, count in counts.most_common()]
    return "\n".join(lines)
