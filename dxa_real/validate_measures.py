"""Поверка измерителя: соответствуем ли требованию ТЗ к погрешности.

ТЗ, п. 3.1: «Погрешность измерения для линейных и объёмных величин не более 5% от экспертной
разметки. Для измерения угловых величин допустимо отклонение не более 3 градусов.»

Проверить это без экспертной разметки координат можно так: взять реальный снимок, изменить
его на ИЗВЕСТНУЮ величину (повернуть на заданный угол, обрезать заданное число миллиметров)
и сравнить измеренное изменение с заданным. Это поверка прибора эталоном.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .data import Image, load_dataset
from .features import (FEMUR_LEVEL, SPINE_LEVEL, bone_mask, largest_component, normalize,
                       rotation_geometry, spine_midline_angle)

ANGLES = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
CROPS_MM = (5.0, 10.0, 15.0, 20.0)


def _measure_angle(array: np.ndarray, spacing, signed: bool = True) -> float:
    mask = bone_mask(normalize(array), SPINE_LEVEL)
    return spine_midline_angle(mask, spacing, signed=signed)[0]


def _measure_femur(array: np.ndarray, spacing, key: str) -> float:
    mask = largest_component(bone_mask(normalize(array), FEMUR_LEVEL), prefer_bottom=True)
    return rotation_geometry(mask, spacing)[key]


def check_angles(images: list[Image], count: int = 20, rng=None) -> dict:
    """Поворачиваем снимок на известный угол и смотрим, насколько изменится измеренный."""
    rng = rng or np.random.default_rng(0)
    spine = [im for im in images if im.region == "spine"]
    picked = [spine[i] for i in rng.choice(len(spine), min(count, len(spine)), replace=False)]
    errors, rows = [], []
    for image in picked:
        base = _measure_angle(image.array, image.spacing)   # знаковый собственный наклон
        for angle in ANGLES:
            for direction in (1.0, -1.0):
                delta = direction * angle
                rotated = ndimage.rotate(image.array.astype(np.float32), delta, reshape=False,
                                         order=1, mode="nearest")
                measured = _measure_angle(rotated, image.spacing)
                expected = base + delta      # поворот складывается с собственным наклоном
                errors.append(abs(measured - expected))
                rows.append((delta, base, measured, measured - expected))
    errors = np.array(errors)
    return dict(n=len(errors), mean_error=float(errors.mean()), median_error=float(np.median(errors)),
                p95_error=float(np.percentile(errors, 95)),
                within_3deg=float((errors <= 3.0).mean()), rows=rows)


def check_lengths(images: list[Image], count: int = 20, rng=None,
                  key: str = "femur_shaft_width_mm") -> dict:
    """Обрезаем кадр снизу на известное число миллиметров и смотрим, устоит ли измерение.

    Размер кости от обрезки кадра меняться не должен: это проверка устойчивости измерения
    и правильности перевода пикселей в миллиметры.
    """
    rng = rng or np.random.default_rng(1)
    femur = [im for im in images if im.region == "femur"]
    picked = [femur[i] for i in rng.choice(len(femur), min(count, len(femur)), replace=False)]
    relative = []
    for image in picked:
        base = _measure_femur(image.array, image.spacing, key)
        if base <= 0:
            continue
        for crop_mm in CROPS_MM:
            rows_to_cut = int(crop_mm / image.spacing[1])
            if rows_to_cut >= image.array.shape[0] * 0.3:
                continue
            cropped = image.array[: image.array.shape[0] - rows_to_cut]   # обрезаем снизу
            measured = _measure_femur(cropped, image.spacing, key)
            if measured > 0:
                relative.append(abs(measured - base) / base)
    relative = np.array(relative)
    return dict(n=len(relative), mean_error=float(relative.mean()),
                median_error=float(np.median(relative)),
                p95_error=float(np.percentile(relative, 95)),
                within_5pct=float((relative <= 0.05).mean()))


def report(dataset_root: str = r"C:\dxa") -> dict:
    images = load_dataset(dataset_root)
    angles = check_angles(images)
    lengths = check_lengths(images, key="femur_shaft_width_mm")
    neck = check_lengths(images, key="femur_neck_width_mm")

    print("ПОВЕРКА ИЗМЕРИТЕЛЯ (требование ТЗ п. 3.1)\n")
    print(f"Углы: {angles['n']} проверок на известных поворотах")
    print(f"   средняя ошибка {angles['mean_error']:.2f}°, медиана {angles['median_error']:.2f}°, "
          f"95-й перцентиль {angles['p95_error']:.2f}°")
    print(f"   в пределах допуска 3°: {angles['within_3deg']:.1%}")
    for name, block in (("ширина диафиза", lengths), ("ширина шейки", neck)):
        print(f"\nЛинейная величина «{name}»: {block['n']} проверок на обрезке кадра")
        print(f"   средняя ошибка {block['mean_error']:.1%}, медиана {block['median_error']:.1%}, "
              f"95-й перцентиль {block['p95_error']:.1%}")
        print(f"   в пределах допуска 5%: {block['within_5pct']:.1%}")
    return dict(angles=angles, lengths=lengths, neck=neck)


if __name__ == "__main__":
    report()
