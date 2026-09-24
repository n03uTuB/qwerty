# -*- coding: utf-8 -*-
"""Атлас-регистрация: выделение кости и поиск «лишнего» вне неё.

Идея (по запросу): DXA — это кость на почти чёрном фоне. Если выделить область
кости и привести снимки к общему виду (центр + масштаб + наклон оси), то у
правильных снимков кость ложится в одно место, а всё, что выбивается за пределы
костного шаблона — посторонние предметы, наложения, артефакты.

Модуль даёт два семейства признаков:

  * **артефакты** (``art_*``) — яркие объекты вне кости, резкие края, тёмные
    включения внутри тела. Это «то, чего не должно быть»;
  * **атлас/укладка** (``atlas_*``, ``align_*``) — насколько снимок совпал с
    шаблоном после регистрации: остаток вне кости, остаток по кости, IoU костной
    маски, потребовавшийся наклон/масштаб. Плохая укладка -> большой наклон и
    расхождение кости с шаблоном.

Шаблон строится БЕЗ учительских меток (медиана зарегистрированных костных масок
и яркостей по реальным снимкам), поэтому утечки метки нет. Артефакт: ``atlas.npz``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from . import config as C
from .features import _as01, bone_mask, largest_component

ATLAS_SIZE = 128
_ATLAS_PATH = os.path.join(C.ARTIFACTS_DIR, "atlas.npz")
_FEMUR_ATLAS_PATH = os.path.join(C.ARTIFACTS_DIR, "atlas_femur.npz")

SPINE_BONE_LEVEL = 0.25
FEMUR_BONE_LEVEL = 0.15


def atlas_path(region: str = "spine") -> str:
    """Путь к файлу шаблона для региона (spine/femur)."""
    return _FEMUR_ATLAS_PATH if region == "femur" else _ATLAS_PATH


def _region_bone_level(region: str) -> float:
    return FEMUR_BONE_LEVEL if region == "femur" else SPINE_BONE_LEVEL


# --------------------------------------------------------------------------- #
# Регистрация
# --------------------------------------------------------------------------- #
def _bone_axis_angle(mask: np.ndarray) -> float:
    """Знаковый наклон главной оси костной маски от вертикали (градусы)."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return 0.0
    x = xs.astype(np.float64)
    y = ys.astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    cov = np.array([[np.mean(x * x), np.mean(x * y)],
                    [np.mean(x * y), np.mean(y * y)]])
    vals, vecs = np.linalg.eigh(cov)
    main = vecs[:, int(np.argmax(vals))]
    if main[1] < 0:
        main = -main
    # угол от вертикали; знак сохраняем (наклон влево/вправо)
    return float(np.degrees(np.arctan2(main[0], abs(main[1]) + 1e-9)))


def _register(arr01: np.ndarray, mask: np.ndarray, size: int = ATLAS_SIZE
              ) -> Optional[Tuple[np.ndarray, np.ndarray, dict]]:
    """Привести снимок к общему виду: центр кости + масштаб + выпрямление оси.

    Возвращает (crop_img, crop_mask, params) или None, если кость не найдена.
    """
    import cv2

    ys, xs = np.nonzero(mask)
    if len(xs) < 40:
        return None
    cy, cx = float(ys.mean()), float(xs.mean())
    angle = _bone_axis_angle(mask)
    # выпрямляем: поворачиваем изображение на -angle вокруг центра кости
    m = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    h, w = arr01.shape
    img_r = cv2.warpAffine(arr01.astype(np.float32), m, (w, h),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask_r = cv2.warpAffine(mask.astype(np.uint8), m, (w, h),
                            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    ys2, xs2 = np.nonzero(mask_r)
    if len(xs2) < 40:
        return None
    y0, y1 = int(ys2.min()), int(ys2.max())
    x0, x1 = int(xs2.min()), int(xs2.max())
    # окно = 2.2 * протяжённость кости, чтобы вокруг кости остался «фон»
    span = max(y1 - y0, x1 - x0)
    half = max(int(span * 1.1), 24)
    top, left = int(cy) - half, int(cx) - half
    win = 2 * half
    # вырезаем с padding (BORDER_CONSTANT=0 — как чёрный фон)
    pad = half
    img_p = cv2.copyMakeBorder(img_r, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    mask_p = cv2.copyMakeBorder(mask_r, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    ty, tx = top + pad, left + pad
    img_c = img_p[ty:ty + win, tx:tx + win]
    mask_c = mask_p[ty:ty + win, tx:tx + win]
    if img_c.shape[0] < 8 or img_c.shape[1] < 8:
        return None
    img_c = cv2.resize(img_c, (size, size), interpolation=cv2.INTER_AREA)
    mask_c = cv2.resize(mask_c.astype(np.float32), (size, size),
                        interpolation=cv2.INTER_NEAREST)
    params = dict(angle=angle, scale=float(span), centroid=(cy, cx))
    return img_c.astype(np.float32), mask_c.astype(np.float32), params


def _spine_mask(arr01: np.ndarray) -> np.ndarray:
    """Костная маска позвоночника (крупнейшая связная компонента)."""
    return largest_component(bone_mask(arr01, SPINE_BONE_LEVEL), prefer_bottom=False)


def _femur_mask(arr01: np.ndarray) -> np.ndarray:
    """Костная маска бедра: крупнейшая компонента, тяготеющая к низу кадра."""
    return largest_component(bone_mask(arr01, FEMUR_BONE_LEVEL), prefer_bottom=True)


def _region_mask(arr01: np.ndarray, region: str) -> np.ndarray:
    """Костная маска для региона (spine/femur)."""
    return _femur_mask(arr01) if region == "femur" else _spine_mask(arr01)


# --------------------------------------------------------------------------- #
# Шаблон
# --------------------------------------------------------------------------- #
@dataclass
class Atlas:
    template: np.ndarray      # (S, S) медиана зарегистрированных яркостей
    bone_prob: np.ndarray     # (S, S) доля снимков, где пиксель — кость
    size: int = ATLAS_SIZE

    def save(self, path: str = _ATLAS_PATH) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(path, template=self.template,
                            bone_prob=self.bone_prob, size=self.size)

    @staticmethod
    def load(path: str = _ATLAS_PATH) -> Optional["Atlas"]:
        if not os.path.isfile(path):
            return None
        d = np.load(path)
        return Atlas(template=d["template"], bone_prob=d["bone_prob"],
                     size=int(d["size"]))


def build_atlas(manifest, path: Optional[str] = None, size: int = ATLAS_SIZE,
                min_count: int = 20, max_images: Optional[int] = None,
                region: str = "spine") -> Optional[Atlas]:
    """Построить шаблон по реальным снимкам региона (без меток).

    region="spine" — позвоночник (99 снимков), region="femur" — проксимальное
    бедро (153 снимка). Шаблон = медиана зарегистрированных яркостей + карта
    вероятности кости.
    """
    from .features import _load_for_features

    if path is None:
        path = atlas_path(region)
    if region == "femur":
        regions = [C.REGION_FEMUR_LEFT, C.REGION_FEMUR_RIGHT]
    else:
        regions = [C.REGION_SPINE]
    rows = manifest[manifest["region"].isin(regions)]
    if "synthetic" in rows:
        rows = rows[~rows["synthetic"].fillna(False).astype(bool)]
    imgs, masks = [], []
    for _, r in rows.iterrows():
        try:
            arr = _load_for_features(r)
            arr01 = _as01(arr)
            mask = _region_mask(arr01, region)
            reg = _register(arr01, mask, size=size)
        except Exception:
            reg = None
        if reg is None:
            continue
        imgs.append(reg[0])
        masks.append(reg[1])
        if max_images is not None and len(imgs) >= max_images:
            break
    if len(imgs) < min_count:
        print(f"[atlas:{region}] мало снимков ({len(imgs)} < {min_count}) — пропуск")
        return None
    stack = np.stack(imgs, axis=0)
    mstack = np.stack(masks, axis=0)
    atlas = Atlas(template=np.median(stack, axis=0).astype(np.float32),
                  bone_prob=mstack.mean(axis=0).astype(np.float32), size=size)
    atlas.save(path)
    print(f"[atlas:{region}] шаблон по {len(imgs)} снимкам -> {path}")
    return atlas


# --------------------------------------------------------------------------- #
# Признаки
# --------------------------------------------------------------------------- #
def artifact_features(arr: np.ndarray, spacing=C.PIXEL_SPACING_MM) -> dict:
    """«Лишнее» вне кости: яркие объекты, резкие края, тёмные включения.

    Работает и без атласа — по одной костной маске. Ключ к критерию «предметы»:
    на DXA фон почти чёрный, поэтому любой яркий объект ВНЕ костной маски —
    кандидат в посторонние предметы/металл.
    """
    from scipy import ndimage

    a = _as01(arr)
    h, w = a.shape
    mask = _spine_mask(a)
    # зона анатомии = кость, слегка расширенная (запас на контур кости)
    anat = ndimage.binary_dilation(mask, np.ones((7, 7)))
    outside = ~anat
    # тело пациента (не чёрный фон)
    body = a > 0.12

    # --- яркие объекты вне кости ---
    # максимум/верхний процентиль яркости вне анатомии: на DXA фон почти чёрный,
    # поэтому яркий пиксель вне кости — почти всегда посторонний предмет (металл)
    if outside.any():
        vals_out = a[outside]
        max_out = float(vals_out.max())
        p999_out = float(np.percentile(vals_out, 99.9))
    else:
        max_out = p999_out = 0.0
    hi = float(np.percentile(a, 99.0))
    bright = (a >= max(hi, 0.55)) & outside
    bright_frac = float(bright.mean())
    lab, n = ndimage.label(bright)
    max_cc = 0.0
    n_cc = 0
    edge_mass = 0.0
    if n > 0:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        max_cc = float(sizes.max()) / (h * w)
        n_cc = int((sizes > 0.0005 * h * w).sum())
        border = np.zeros_like(bright)
        b = max(2, int(0.06 * min(h, w)))
        border[:b, :] = border[-b:, :] = True
        border[:, :b] = border[:, -b:] = True
        edge_mass = float((bright & border).sum()) / max(float(bright.sum()), 1.0)

    # --- резкие края вне кости (металл даёт сильный градиент) ---
    gx = ndimage.sobel(a, axis=1)
    gy = ndimage.sobel(a, axis=0)
    grad = np.hypot(gx, gy)
    edge_out = float(grad[outside].mean()) if outside.any() else 0.0
    edge_out_p99 = float(np.percentile(grad[outside], 99.0)) if outside.any() else 0.0
    edge_out_p999 = float(np.percentile(grad[outside], 99.9)) if outside.any() else 0.0

    # --- тёмные включения внутри тела (низкоплотные зоны) ---
    lo = float(np.percentile(a[body], 5.0)) if body.any() else 0.0
    dark = (a <= lo) & anat & body
    dark_frac = float(dark.mean())

    return {
        "art_max_out": max_out,
        "art_p999_out": p999_out,
        "art_bright_out_frac": bright_frac,
        "art_bright_out_maxcc": max_cc,
        "art_bright_out_ncc": float(n_cc),
        "art_bright_out_edge": edge_mass,
        "art_edge_out_mean": edge_out,
        "art_edge_out_p99": edge_out_p99,
        "art_edge_out_p999": edge_out_p999,
        "art_dark_in_frac": dark_frac,
    }


def atlas_features(arr: np.ndarray, atlas: Optional[Atlas],
                   spacing=C.PIXEL_SPACING_MM, region: str = "spine") -> dict:
    """Остаток относительно костного шаблона + параметры регистрации.

    region="spine" -> имена ``atlas_*``/``align_*``; region="femur" ->
    ``fatlas_*``/``falign_*`` (отдельные шаблон и маска).
    """
    pfx = "fatlas_" if region == "femur" else "atlas_"
    apfx = "falign_" if region == "femur" else "align_"
    zeros = {
        pfx + "resid_out": 0.0, pfx + "resid_out_p99": 0.0,
        pfx + "outlier_area": 0.0, pfx + "resid_bone": 0.0,
        pfx + "bone_iou": 0.0, apfx + "angle": 0.0, apfx + "scale": 0.0,
    }
    if atlas is None:
        return zeros
    a = _as01(arr)
    mask = _region_mask(a, region)
    reg = _register(a, mask, size=atlas.size)
    if reg is None:
        return zeros
    img_c, mask_c, params = reg
    resid = np.abs(img_c - atlas.template)
    bone = atlas.bone_prob
    out_zone = bone < 0.2
    in_zone = bone > 0.5

    resid_out = float(resid[out_zone].mean()) if out_zone.any() else 0.0
    resid_out_p99 = float(np.percentile(resid[out_zone], 99.0)) if out_zone.any() else 0.0
    # площадь «выбросов» вне кости: насколько снимок «загрязнён» вне шаблона
    tau = 0.35
    outlier_area = float(((resid > tau) & out_zone).mean())
    resid_bone = float(resid[in_zone].mean()) if in_zone.any() else 0.0
    # IoU костной маски снимка и шаблонной (мера совпадения укладки)
    this_bone = mask_c > 0.5
    tmpl_bone = bone > 0.5
    inter = float((this_bone & tmpl_bone).sum())
    union = float((this_bone | tmpl_bone).sum())
    iou = inter / union if union > 0 else 0.0

    return {
        pfx + "resid_out": resid_out,
        pfx + "resid_out_p99": resid_out_p99,
        pfx + "outlier_area": outlier_area,
        pfx + "resid_bone": resid_bone,
        pfx + "bone_iou": iou,
        apfx + "angle": abs(float(params["angle"])),
        apfx + "scale": float(params["scale"]),
    }


def main():
    """CLI: построить атлас-шаблон -> artifacts/atlas.npz."""
    import argparse

    from .data import load_manifest

    ap = argparse.ArgumentParser(description="Построение костного атлас-шаблона")
    ap.add_argument("--manifest", default=C.MANIFEST_CSV)
    ap.add_argument("--size", type=int, default=ATLAS_SIZE)
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--region", default="both",
                    choices=["spine", "femur", "both"])
    args = ap.parse_args()
    manifest = load_manifest(args.manifest)
    regions = (["spine", "femur"] if args.region == "both" else [args.region])
    for reg in regions:
        build_atlas(manifest, size=args.size, min_count=args.min_count,
                    region=reg)


if __name__ == "__main__":
    main()
