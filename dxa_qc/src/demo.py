# -*- coding: utf-8 -*-
"""Демонстрация: end-to-end прогон по датасету и сравнение с разметкой.

Запуск:
    python -m src.demo --input <НД_для_обучения> --limit 8
Создаёт outputs/demo_results.csv и outputs/demo_gradcam/*.png.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch

from . import config as C
from . import dataset as ds
from . import dicom_io
from . import inference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=C.DEFAULT_DATASET_ROOT)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gradcam", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    controller = inference.build_controller_from_args(device=args.device)
    print(f"[demo] моделей в ансамбле: {len(controller.models)}")

    manifest = ds.load_manifest(C.MANIFEST_CSV)
    studies = list(dict.fromkeys(manifest["study_uid"].tolist()))[: args.limit]

    studies_root = os.path.join(args.input, "исследования")
    if not os.path.isdir(studies_root):
        studies_root = args.input

    rows = []
    for uid in studies:
        sp = os.path.join(studies_root, uid)
        if not os.path.isdir(sp):
            continue
        rows.extend(inference.process_study(controller, uid, sp))

    out = pd.DataFrame(rows)
    os.makedirs(C.OUTPUTS_DIR, exist_ok=True)
    out_path = os.path.join(C.OUTPUTS_DIR, "demo_results.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[demo] результаты: {out_path}  строк: {len(out)}")

    # сопоставление с истинной разметкой
    merged = out.merge(
        manifest[["study_uid", "region", "quality"]].rename(
            columns={"region": "true_region", "quality": "true_quality"}),
        on=["study_uid"], how="left")
    print("\n[demo] предсказание vs разметка (по области):")
    show = out[["study_uid", "anatomical_region", "quality_class",
                "violation_type", "processing_status"]].copy()
    show["study_uid"] = show["study_uid"].str[:20]
    print(show.to_string(index=False))

    if args.gradcam:
        from . import visualize

        gdir = os.path.join(C.OUTPUTS_DIR, "demo_gradcam")
        os.makedirs(gdir, exist_ok=True)
        n = 0
        for uid in studies:
            sp = os.path.join(studies_root, uid)
            files = dicom_io.find_dicom_files(sp)
            uniq = dicom_io.dedupe_unique_images(files)
            for path, dset, arr in uniq[:1]:
                out_png = os.path.join(gdir, f"{uid[:24]}_{n}.png")
                try:
                    visualize.save_overlay(controller, path, out_png, head="quality")
                    n += 1
                except Exception as e:
                    print("  [warn] gradcam:", e)
        print(f"[demo] тепловые карты: {gdir}")


if __name__ == "__main__":
    main()
