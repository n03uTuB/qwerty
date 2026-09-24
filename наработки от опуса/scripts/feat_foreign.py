# -*- coding: utf-8 -*-
"""Прямые измерения критерия «посторонние предметы» для spine_artifacts.

Порт `foreign_body()` из dxa_real/features.py (в dxa_qc он не используется и в
features.csv отсутствует) с усилениями:
  * асимметрия слева/справа — артефакт обычно односторонний;
  * интенсивность блоба к плотности кости (p98) — «плотнее кости» из ТЗ;
  * вытянутость блобов — линейные наложения (молнии/застёжки).

Изображение читается ровно так же, как при построении features.csv (сырой DICOM
для реальных, кэш PNG для синтетики), чтобы признаки были согласованы.

Запуск:
    python dxa_qc_work/scripts/feat_foreign.py
Результат: dxa_qc_work/out/feat_foreign.csv
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import _common  # noqa: F401

from src import config as C        # noqa: E402
from src import features as fl     # noqa: E402

FOREIGN_COLS = [
    "foreign_count", "foreign_area", "foreign_max_area", "foreign_compactness",
    "foreign_top", "foreign_peak_rel", "foreign_asym", "foreign_elong_max",
    "foreign_elong_count", "foreign_area_left", "foreign_area_right",
]

LEVEL = 0.75
MAX_AREA = 0.03
MIN_SIZE = 15


def _empty() -> dict:
    return {c: 0.0 for c in FOREIGN_COLS}


def foreign_features(values: np.ndarray) -> dict:
    from scipy import ndimage

    values = np.asarray(values).astype(np.float32)
    soft, dense = np.percentile(values, (50, 98))
    if dense <= soft:
        return _empty()
    bright = values > soft + (dense - soft) * LEVEL
    skeleton = fl.largest_component(fl.bone_mask(values, fl.SPINE_LEVEL))
    outside = bright & ~ndimage.binary_dilation(skeleton, np.ones((9, 9)))
    labels, count = ndimage.label(ndimage.binary_opening(outside, np.ones((2, 2))))
    if not count:
        return _empty()

    rows, cols = values.shape
    total = values.size
    found = []
    for index in range(1, count + 1):
        ys, xs = np.nonzero(labels == index)
        size = len(ys)
        if size < MIN_SIZE or size > total * MAX_AREA:
            continue
        box = max((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1), 1)
        # вытянутость по вторым моментам
        cy, cx = ys.mean(), xs.mean()
        cov = np.cov(np.stack([xs - cx, ys - cy])) if size > 2 else np.eye(2)
        ev = np.linalg.eigvalsh(cov)
        elong = float(np.sqrt(max(ev.max(), 1e-6) / max(ev.min(), 1e-6)))
        found.append(dict(
            size=size,
            compactness=size / box,
            top=1.0 - cy / rows,
            x_center=cx / cols,
            mean_intensity=float(values[ys, xs].mean()),
            elong=elong,
        ))
    if not found:
        return _empty()

    left = sum(f["size"] for f in found if f["x_center"] < 0.5)
    right = sum(f["size"] for f in found if f["x_center"] >= 0.5)
    return dict(
        foreign_count=float(len(found)),
        foreign_area=float(sum(f["size"] for f in found) / total),
        foreign_max_area=float(max(f["size"] for f in found) / total),
        foreign_compactness=float(max(f["compactness"] for f in found)),
        foreign_top=float(max(f["top"] for f in found)),
        foreign_peak_rel=float(max(f["mean_intensity"] for f in found) / max(dense, 1e-6)),
        foreign_asym=float(abs(left - right) / max(left + right, 1)),
        foreign_elong_max=float(max(f["elong"] for f in found)),
        foreign_elong_count=float(sum(1 for f in found if f["elong"] > 3.0)),
        foreign_area_left=float(left / total),
        foreign_area_right=float(right / total),
    )


def main():
    out_csv = os.path.join(_common.OUT, "feat_foreign.csv")
    df = pd.read_csv(C.MANIFEST_CSV)
    rows = []
    for i, r in df.iterrows():
        try:
            arr = fl._load_for_features(r)
            feat = foreign_features(arr)
        except Exception as exc:  # noqa: BLE001
            print(f"  ошибка на {r['image_uid']}: {exc}")
            feat = _empty()
        feat["image_uid"] = r["image_uid"]
        rows.append(feat)
        if (i + 1) % 50 == 0:
            print(f"  обработано {i + 1}/{len(df)}")
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\nсохранено: {out_csv}  строк: {len(out)}")
    print(out[FOREIGN_COLS].describe().to_string())


if __name__ == "__main__":
    main()
