# -*- coding: utf-8 -*-
"""CLI пакетной обработки: python -m src.predict --input <dir> --output <csv>."""
from __future__ import annotations

import argparse
import os

import torch

from . import config as C
from . import inference


def main():
    ap = argparse.ArgumentParser(description="DXA-QC пакетный инференс")
    ap.add_argument("--input", required=True, help="каталог с исследованиями "
                    "(либо корень НД_для_обучения, либо каталог 'исследования')")
    ap.add_argument("--output", default=os.path.join(C.OUTPUTS_DIR, "results.csv"))
    ap.add_argument("--xlsx", default=None)
    ap.add_argument("--weights", nargs="*", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--quality-threshold", type=float, default=None)
    ap.add_argument("--violation-threshold", type=float, default=None)
    ap.add_argument("--backbone", default=None,
                    help="бэкбон (resnet18/resnet34/...). По умолчанию "
                         "определяется автоматически по чекпоинту.")
    ap.add_argument("--size", type=int, default=C.IMAGE_SIZE)
    args = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() or 1))

    controller = inference.build_controller_from_args(
        weights=args.weights, device=args.device,
        quality_threshold=args.quality_threshold,
        violation_threshold=args.violation_threshold, size=args.size,
        backbone=args.backbone)
    print(f"[predict] device={args.device} моделей: {len(controller.models)} "
          f"бэкбоны={getattr(controller, 'backbones', None)} "
          f"q_thr(общий)={controller.quality_threshold:.3f} "
          f"q_thr(по областям)={controller.quality_thresholds}")

    xlsx = args.xlsx
    if xlsx is None:
        xlsx = os.path.splitext(args.output)[0] + ".xlsx"
    df = inference.run_batch(args.input, args.output, controller, output_xlsx=xlsx)
    n_ok = int((df.processing_status == "Success").sum())
    print(f"[predict] всего строк: {len(df)}  успешно: {n_ok}  ->  {args.output}")


if __name__ == "__main__":
    main()
