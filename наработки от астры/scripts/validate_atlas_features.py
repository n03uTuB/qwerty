# -*- coding: utf-8 -*-
"""Валидация новых признаков (art_* и atlas_*) на синтетическом стенде.

Реальных данных на этой машине нет, поэтому стенд строит правдоподобные
DXA-подобные изображения (яркая кость на чёрном фоне) и проверяет, разделяют ли
признаки:

  1. «предметы»   — снимок с наложенным посторонним объектом vs чистый;
  2. «укладка»    — сдвинутая/повёрнутая кость vs ровная.

Считается ROC-AUC каждого признака в отдельности (без обучения) — это честная
оценка «есть ли сигнал», а не подгонка. Запуск:

    python scripts/validate_atlas_features.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config as C  # noqa: E402
from src.atlas import Atlas, artifact_features, atlas_features, build_atlas  # noqa: E402
from src.data import synthetic as syn  # noqa: E402
from src.features import compute_features  # noqa: E402

RNG = np.random.default_rng(0)


def base_spine(rng, rows=300, cols=320, shift=0.0, angle=0.0) -> np.ndarray:
    """Правдоподобный позвоночник: яркий столб на чёрном фоне (+ мягкие ткани)."""
    import cv2

    img = np.zeros((rows, cols), np.float32)
    cx = cols // 2 + int(shift * cols)
    # мягкие ткани — широкий слабый овал
    yy, xx = np.mgrid[0:rows, 0:cols]
    body = (((xx - cx) / (cols * 0.42)) ** 2 + ((yy - rows / 2) / (rows * 0.55)) ** 2) < 1
    img[body] = rng.uniform(0.10, 0.20)
    # позвоночный столб — узкий яркий столб
    col = (np.abs(xx - cx) < cols * 0.055) & (yy > rows * 0.12) & (yy < rows * 0.92)
    img[col] = rng.uniform(0.75, 0.95)
    # гребни подвздошных костей — ярче внизу
    iliac = (np.abs(xx - cx) < cols * 0.30) & (yy > rows * 0.82)
    img[iliac] = np.maximum(img[iliac], rng.uniform(0.55, 0.75))
    # тела позвонков — «ступеньки» по столбу
    for k in range(5):
        y0 = int(rows * (0.18 + 0.14 * k))
        img[y0:y0 + 8, cx - int(cols * 0.075):cx + int(cols * 0.075)] = rng.uniform(0.85, 1.0)
    img = np.clip(img + rng.normal(0, 0.01, img.shape), 0, 1)
    if angle != 0.0:
        m = cv2.getRotationMatrix2D((cx, rows / 2), angle, 1.0)
        img = cv2.warpAffine(img, m, (cols, rows), borderMode=cv2.BORDER_CONSTANT)
    return (img * 255).astype(np.uint8)


def _auc(y, x):
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, x))
    except Exception:
        return float("nan")


def main():
    n = 40
    clean, art, mis = [], [], []
    for i in range(n):
        b = base_spine(RNG)
        clean.append(b)
        art.append(syn.edge_object(b, RNG))
        mis.append(base_spine(RNG, shift=RNG.uniform(0.06, 0.14),
                              angle=RNG.uniform(6, 14)))
    print("стенд: clean=%d  artifacts=%d  misposition=%d" % (n, n, n))

    # --- признаки артефактов ---
    art_feats = [k for k in artifact_features(clean[0]).keys()]
    rows_clean = [artifact_features(x) for x in clean]
    rows_art = [artifact_features(x) for x in art]
    y = np.array([0] * n + [1] * n)
    print("\n=== критерий «предметы»: AUC признака (artifact vs clean) ===")
    for f in art_feats:
        x = np.array([r[f] for r in rows_clean] + [r[f] for r in rows_art])
        print("  %-24s AUC=%.3f" % (f, _auc(y, x)))

    # --- атлас: построить по чистым, затем признаки на clean vs misposition ---
    atlas = Atlas(template=np.median(
        np.stack([_reg(x)[0] for x in clean]), axis=0),
        bone_prob=np.mean(np.stack([_reg(x)[1] for x in clean]), axis=0))
    print("\n=== «укладка»: AUC признака (misposition vs clean) ===")
    fc = [atlas_features(x, atlas) for x in clean]
    fm = [atlas_features(x, atlas) for x in mis]
    for f in fc[0].keys():
        x = np.array([r[f] for r in fc] + [r[f] for r in fm])
        print("  %-24s AUC=%.3f" % (f, _auc(y, x)))


def _reg(img8):
    from src.atlas import _register, _spine_mask
    from src.features import _as01
    a = _as01(img8)
    return _register(a, _spine_mask(a))


if __name__ == "__main__":
    main()