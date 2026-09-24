# -*- coding: utf-8 -*-
"""Добавить синтетические «предметы/артефакты/наложения» v2 в отдельный манифест.

Берёт текущий манифест dxa_qc и ДОПИСЫВАЕТ к нему синтетику артефактов, сохраняя
результат в отдельный файл (по умолчанию out/manifest_art2.csv), чтобы можно было
честно сравнить обучение с ней и без неё, не трогая рабочий манифест.

Запуск:
    python scripts/make_artifacts_synth.py --dataset C:\\dxa --per-type 60
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dxa_real import synth_artifacts
from dxa_real.data import LABEL_FOREIGN, load_dataset

from dxa_qc.src import config as C  # noqa: E402

LABEL_TO_COLUMN = {LABEL_FOREIGN: {"spine": "spine_artifacts"}}
VIOLATIONS = ["spine_positioning", "spine_axis", "spine_artifacts",
              "femur_positioning", "femur_roi"]
REGION_TO_IDX = {"lumbar_spine": 0, "proximal_femur_left": 1, "proximal_femur_right": 2}


def region_name(image) -> str:
    if image.region == "spine":
        return "lumbar_spine"
    return "proximal_femur_left" if image.side == "l" else "proximal_femur_right"


def normalize_uint8(array: np.ndarray) -> np.ndarray:
    values = array.astype(np.float32)
    lo, hi = np.percentile(values, (0.5, 99.5))
    return (np.clip((values - lo) / max(hi - lo, 1e-6), 0, 1) * 255).round().astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--manifest", default=C.MANIFEST_CSV)
    ap.add_argument("--out", default=str(Path("out") / "manifest_art2.csv"))
    ap.add_argument("--per-type", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    print(f"[манифест] строк всего: {len(manifest)} "
          f"(реальных: {int((~manifest.get('synthetic', False).fillna(False)).sum())})")

    images = load_dataset(args.dataset)
    generated = synth_artifacts.build_artifacts(
        images, np.random.default_rng(args.seed), per_type=args.per_type)

    # убираем точные дубликаты (тот же источник + подтип)
    unique, seen = [], set()
    for image in generated:
        if image.image_uid in seen:
            continue
        seen.add(image.image_uid)
        unique.append(image)
    if len(unique) != len(generated):
        print(f"[синтетика] отброшено дубликатов: {len(generated) - len(unique)}")
    generated = unique
    print(synth_artifacts.summary(generated))

    cache_dir = Path(C.CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for image in generated:
        region = region_name(image)
        digest = hashlib.md5(image.image_uid.encode("utf-8")).hexdigest()
        path = cache_dir / f"synth_{digest}.png"
        PILImage.fromarray(normalize_uint8(image.array)).save(path)

        row = {
            "study_uid": image.study_uid,
            "image_uid": image.image_uid,
            "source_path": str(image.path),
            "cache_path": str(path.resolve()),
            "region": region,
            "region_idx": REGION_TO_IDX[region],
            "rows": int(image.array.shape[0]),
            "cols": int(image.array.shape[1]),
            "spacing_x": image.spacing[0],
            "spacing_y": image.spacing[1],
            "copies": 1,
            "quality": float(image.quality),
            "synthetic": True,
        }
        for name in VIOLATIONS:
            row[f"viol_{name}"] = np.nan
            row[f"mask_{name}"] = 0.0
        for label, value in image.labels.items():
            column = LABEL_TO_COLUMN.get(label, {}).get(image.region)
            if column:
                row[f"viol_{column}"] = float(value)
                row[f"mask_{column}"] = 1.0
        rows.append(row)

    combined = pd.concat([manifest, pd.DataFrame(rows)], ignore_index=True)
    if "synthetic" in combined:
        combined["synthetic"] = combined["synthetic"].fillna(False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out, index=False, encoding="utf-8")
    print(f"\n[манифест] сохранён: {args.out}  строк: {len(combined)} "
          f"(добавлено {len(rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
