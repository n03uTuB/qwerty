"""Добавляет синтетические нарушения в пайплайн dxa_qc сокомандника.

Что делает:
  1. читает набор организатора и генерирует нарушения из чистых снимков;
  2. сохраняет их в кэш изображений dxa_qc (artifacts/image_cache/*.png);
  3. дописывает строки в artifacts/manifest.csv с теми же колонками, что у обычных снимков.

Дальше обучение запускается как обычно: синтетика попадёт в обучающую выборку
автоматически. study_uid наследуется от исходного снимка, поэтому GroupKFold
не даст синтетике утечь в валидацию к своему же источнику.

Запуск:
    python scripts/make_synthetic_for_dxa_qc.py --dataset C:\\dxa --project dxa_qc --per-type 40
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

from dxa_real import synth_real
from dxa_real.data import (LABEL_AXIS, LABEL_FOREIGN, LABEL_POSITIONING, LABEL_ROI, load_dataset)

# метка организатора -> колонка критерия в манифесте dxa_qc
LABEL_TO_COLUMN = {
    LABEL_POSITIONING: {"spine": "spine_positioning", "femur": "femur_positioning"},
    LABEL_AXIS: {"spine": "spine_axis"},
    LABEL_FOREIGN: {"spine": "spine_artifacts"},
    LABEL_ROI: {"femur": "femur_roi"},
}
VIOLATIONS = ["spine_positioning", "spine_axis", "spine_artifacts", "femur_positioning", "femur_roi"]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="каталог НД_для_обучения")
    parser.add_argument("--project", default="dxa_qc", help="каталог решения dxa_qc")
    parser.add_argument("--per-type", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    project = Path(args.project)
    manifest_path = project / "artifacts" / "manifest.csv"
    cache_dir = project / "artifacts" / "image_cache"
    if not manifest_path.is_file():
        print(f"нет манифеста {manifest_path}. Сначала соберите его:\n"
              f"  python -c \"from src import dataset; dataset.build_manifest(r'{args.dataset}')\"")
        return 1
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)
    manifest = manifest[~manifest.get("synthetic", pd.Series(False, index=manifest.index)).fillna(False)]
    print(f"[манифест] реальных строк: {len(manifest)}")

    images = load_dataset(args.dataset)
    generated = synth_real.build(images, np.random.default_rng(args.seed), per_type=args.per_type)
    print(synth_real.summary(generated))

    rows = []
    for image in generated:
        region = region_name(image)
        digest = hashlib.md5(image.image_uid.encode("utf-8")).hexdigest()
        path = cache_dir / f"synth_{digest}.png"
        PILImage.fromarray(normalize_uint8(image.array)).save(path)

        row = {
            "study_uid": image.study_uid,          # тот же пациент -> тот же фолд
            "image_uid": image.image_uid,
            "source_path": str(image.path),
            "cache_path": str(path.resolve()),
            "region": region,
            "region_idx": REGION_TO_IDX[region],
            "rows": int(image.array.shape[0]),
            "cols": int(image.array.shape[1]),
            "spacing_x": image.spacing[0],
            "spacing_y": image.spacing[1],
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
    combined.to_csv(manifest_path, index=False, encoding="utf-8")
    print(f"\n[манифест] стало строк: {len(combined)} (добавлено {len(rows)})")
    print("Дальше как обычно:")
    print("  python -m src.train        # синтетика уже в обучающей выборке")
    print("  python -m src.stack && python -m src.calibrate && python -m src.report")
    print("\nВАЖНО: метрики считайте только по реальным снимкам (synthetic == False).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
