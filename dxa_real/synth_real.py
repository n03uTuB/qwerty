"""Синтетические нарушения из РЕАЛЬНЫХ чистых снимков набора.

Зачем: положительных примеров критически мало (6 «укладка позвоночника»,
7 «область интереса», 10 «ось»). Учить на этом нечего. Берём снимки, у которых
эксперт не нашёл нарушений, и портим их контролируемо.

Главный риск — модель выучит следы самого преобразования (чёрные углы от поворота,
сглаживание интерполяцией), а не нарушение. Защита:
  * поворот с заполнением краёв по краю (mode="nearest"), без чёрных клиньев;
  * КОНТРОЛЬНЫЕ НЕГАТИВЫ: те же чистые снимки проходят ту же обработку с малым
    углом и остаются с меткой 0, поэтому «след поворота» есть у обоих классов;
  * синтетика наследует study_uid источника и потому остаётся в той же части
    кросс-валидации — оценка идёт только по реальным снимкам.
"""

from __future__ import annotations

import copy

import numpy as np
from scipy import ndimage

from .data import (LABEL_AXIS, LABEL_FOREIGN, LABEL_POSITIONING, LABEL_ROI, Image)
from .features import bone_mask, normalize

# Целимся в распределение НАСТОЯЩИХ нарушений: у них измеренный наклон 3.4–9.1°,
# медиана 5.0°. Слишком сильный поворот сделал бы задачу легче реальной.
AXIS_TARGET_DEG = (4.5, 9.0)
CONTROL_DEG = (0.5, 2.0)           # контрольный поворот: нарушения нет


def _derive(image: Image, array: np.ndarray, suffix: str, labels: dict[str, int]) -> Image:
    clone = copy.copy(image)
    clone.array = array
    clone.image_uid = f"{image.image_uid}:{suffix}"
    clone.labels = dict(labels)
    clone.quality = int(any(labels.values()))
    clone.synthetic = True
    clone.source_uid = image.image_uid
    return clone


def is_clean(image: Image) -> bool:
    return bool(image.labels) and image.quality == 0 and not any(image.labels.values())


def _measure_axis(image: Image) -> float:
    """Текущий наклон оси снимка (тем же измерением, что и в признаках)."""
    from .features import SPINE_LEVEL, spine_midline_angle

    return spine_midline_angle(bone_mask(normalize(image.array), SPINE_LEVEL), image.spacing)[0]


def rotate(image: Image, angle: float) -> np.ndarray:
    """Поворот без чёрных углов: края достраиваются краевыми пикселями."""
    return ndimage.rotate(image.array.astype(np.float32), angle, reshape=False, order=1,
                          mode="nearest").clip(0, 255).astype(image.array.dtype)


def rotate_to_angle(image: Image, target_deg: float, measure) -> tuple[np.ndarray, float]:
    """Повернуть так, чтобы ИЗМЕРЕННЫЙ наклон стал равен target_deg.

    Снимок уже имеет собственный небольшой наклон, поэтому поворачиваем на разницу,
    а знак выбираем так, чтобы наклон увеличивался.
    """
    current = measure(image)
    delta = max(target_deg - current, 0.5)
    best = None
    for sign in (1.0, -1.0):  # снимок уже наклонён: в одну сторону угол растёт, в другую падает
        candidate = rotate(image, sign * delta)
        probe = copy.copy(image)
        probe.array = candidate
        error = abs(measure(probe) - target_deg)
        if best is None or error < best[0]:
            best = (error, candidate, sign * delta)
    return best[1], best[2]


def paste_metal(image: Image, rng: np.random.Generator) -> np.ndarray:
    """Яркий компактный объект в мягких тканях: застёжка белья, пуговица, пирсинг."""
    array = image.array.astype(np.float32)
    values = normalize(array)
    outside = ~ndimage.binary_dilation(bone_mask(values, 0.25), np.ones((9, 9)))
    body = values > 0.12                      # тело пациента, а не воздух вокруг
    candidates = np.argwhere(outside & body)
    rows, cols = array.shape
    if len(candidates) < 50:
        cy, cx = rows * 0.15, cols * 0.5
    else:
        cy, cx = candidates[rng.integers(len(candidates))]

    yy, xx = np.mgrid[0:rows, 0:cols]
    ry = rng.uniform(3, 7)
    rx = rng.uniform(3, 9)
    angle = rng.uniform(0, np.pi)
    dy, dx = yy - cy, xx - cx
    rot_y = dy * np.cos(angle) + dx * np.sin(angle)
    rot_x = -dy * np.sin(angle) + dx * np.cos(angle)
    blob = (rot_y / ry) ** 2 + (rot_x / rx) ** 2 <= 1.0
    patch = ndimage.gaussian_filter(blob.astype(np.float32), sigma=0.8)
    target = float(array.max())
    array = array * (1 - patch) + target * patch
    return array.clip(0, 255).astype(image.array.dtype)


def shrink_fov(image: Image, rng: np.random.Generator, where: str,
               resize_back: bool = True) -> np.ndarray:
    """Уменьшение поля сканирования обрезкой края кадра.

    resize_back=True — кадр растягивается до исходного размера (нарушение нельзя
    опознать просто по числу строк). Так делается для укладки позвоночника.

    resize_back=False — кадр остаётся меньше. Так выглядят НАСТОЯЩИЕ нарушения
    области интереса на бедре: у них укороченное поле сканирования, и высота кадра —
    самый сильный реальный признак. С растягиванием синтетика противоречила бы ему.
    """
    array = image.array
    rows, cols = array.shape
    cut = rng.uniform(0.12, 0.25)
    if where == "bottom":
        cropped = array[: int(rows * (1 - cut))]
    elif where == "top":
        cropped = array[int(rows * cut):]
    elif where == "left":
        cropped = array[:, int(cols * cut):]
    else:
        cropped = array[:, : int(cols * (1 - cut))]
    if not resize_back:
        return cropped.copy()
    zoom = (rows / cropped.shape[0], cols / cropped.shape[1])
    return ndimage.zoom(cropped.astype(np.float32), zoom, order=1).clip(0, 255).astype(array.dtype)


# Какие виды генерировать. Проверено экспериментом (см. docs/synthetic.md):
# повороты и обрезка кадра помогают, вклейка металла — вредит, потому что настоящие
# «посторонние предметы и наложения» выглядят иначе, чем яркий компактный объект.
DEFAULT_KINDS = ("axis", "spine_crop", "femur_fov", "control")


def build(images: list[Image], rng: np.random.Generator, per_type: int = 40,
          kinds: tuple[str, ...] = DEFAULT_KINDS) -> list[Image]:
    """Синтетические нарушения плюс контрольные негативы с той же обработкой."""
    spine = [im for im in images if im.region == "spine" and is_clean(im)]
    femur = [im for im in images if im.region == "femur" and is_clean(im)]
    generated: list[Image] = []

    def pick(pool):
        return pool[int(rng.integers(len(pool)))]

    for _ in range(per_type):
        if spine and "axis" in kinds:
            # наклон оси: доводим измеренный угол до уровня реальных нарушений
            source = pick(spine)
            target = float(rng.uniform(*AXIS_TARGET_DEG))
            rotated, angle = rotate_to_angle(source, target, _measure_axis)
            generated.append(_derive(source, rotated, f"rot{angle:+.1f}",
                                     {**{k: 0 for k in source.labels}, LABEL_AXIS: 1}))
        if spine and "spine_crop" in kinds:
            # укладка: обрезан низ, не видны гребни подвздошных костей
            source = pick(spine)
            generated.append(_derive(source, shrink_fov(source, rng, "bottom"), "cut",
                                     {**{k: 0 for k in source.labels}, LABEL_POSITIONING: 1}))
        if spine and "metal" in kinds:
            source = pick(spine)
            generated.append(_derive(source, paste_metal(source, rng), "metal",
                                     {**{k: 0 for k in source.labels}, LABEL_FOREIGN: 1}))
        if femur and "femur_fov" in kinds:
            # область интереса: укороченное поле сканирования, кадр остаётся меньше
            source = pick(femur)
            side = str(rng.choice(["top", "bottom", "left", "right"]))
            generated.append(_derive(source, shrink_fov(source, rng, side, resize_back=False),
                                     f"fov_{side}",
                                     {**{k: 0 for k in source.labels}, LABEL_ROI: 1}))
        if "control" in kinds:
            # контроль: та же обработка (поворот), но в пределах нормы -> метка 0
            for pool in (spine, femur):
                if not pool:
                    continue
                source = pick(pool)
                small = float(rng.choice([-1, 1]) * rng.uniform(*CONTROL_DEG))
                generated.append(_derive(source, rotate(source, small), f"ctl{small:+.1f}",
                                         {k: 0 for k in source.labels}))
    return generated


def summary(generated: list[Image]) -> str:
    from collections import Counter

    counts = Counter()
    for image in generated:
        positive = [label for label, value in image.labels.items() if value]
        counts[positive[0] if positive else "контроль (без нарушений)"] += 1
    lines = [f"синтетических снимков: {len(generated)}"]
    lines += [f"  {name}: {count}" for name, count in counts.most_common()]
    return "\n".join(lines)
