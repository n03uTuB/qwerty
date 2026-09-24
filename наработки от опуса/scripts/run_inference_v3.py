# -*- coding: utf-8 -*-
"""Инференс улучшенной конфигурации v3 на пакете DICOM (mogaem не изменяется).

Собирает самодостаточный runtime-бандл в dxa_qc_work/out/v3/runtime:
  * stacker.joblib  — стекеры v3 (ось с residual, предметы fused);
  * thresholds.json — пороги v3;
  * веса модели     — из --weights-dir (по умолчанию артефакты репозитория).

Затем указывает config.ARTIFACTS_DIR на этот бандл и вызывает существующий
`src.inference_v2` — код репозитория не правится.

Запуск:
    python dxa_qc_work/scripts/run_inference_v3.py --input "...\\_dsroot\\Исследования" \
        --out ..\\out\\v3\\predictions.csv [--device cuda]
"""
from __future__ import annotations

import argparse
import os
import shutil

import _common  # noqa: F401

from src import config as C        # noqa: E402

V3 = os.path.join(_common.OUT, "v3")
RUNTIME = os.path.join(V3, "runtime")


def build_runtime(weights_dir: str) -> str:
    os.makedirs(RUNTIME, exist_ok=True)
    # стекеры и пороги v3
    shutil.copy2(os.path.join(V3, "stackers_v3.joblib"), os.path.join(RUNTIME, "stacker.joblib"))
    shutil.copy2(os.path.join(V3, "thresholds_v3.json"), os.path.join(RUNTIME, "thresholds.json"))
    # веса
    n = 0
    for fn in os.listdir(weights_dir):
        if fn.endswith(".pt"):
            shutil.copy2(os.path.join(weights_dir, fn), os.path.join(RUNTIME, fn))
            n += 1
    print(f"[runtime] скопировано весов: {n} из {weights_dir}")
    return RUNTIME


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default=os.path.join(V3, "predictions.csv"))
    ap.add_argument("--xlsx", default=None)
    ap.add_argument("--weights-dir", default=C.ARTIFACTS_DIR,
                    help="каталог с .pt (по умолчанию — артефакты репозитория)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    build_runtime(args.weights_dir)

    # переключаем каталог артефактов на бандл v3 (стекеры/пороги/веса)
    C.ARTIFACTS_DIR = RUNTIME

    from src import inference_v2
    controller = inference_v2.build_controller(device=args.device)
    print(f"[infer] устройство: {controller.device}  моделей: {len(controller.models)}  "
          f"стекеров: {sorted(controller.stackers)}")
    df = inference_v2.run_batch(args.input, args.out, controller, output_xlsx=args.xlsx)
    print(f"\n[infer] строк: {len(df)}  ->  {args.out}")
    print(df.head(10).to_string())
    statuses = df["processing_status"].value_counts().to_dict() if "processing_status" in df else {}
    print("[infer] статусы:", statuses)


if __name__ == "__main__":
    main()
