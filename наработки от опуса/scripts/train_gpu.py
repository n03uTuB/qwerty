# -*- coding: utf-8 -*-
"""Обучение CNN v2 на GPU. Пишет ВСЁ в dxa_qc_work/out/artifacts (mogaem не трогаем).

Манифест и кэш изображений берутся из репозитория (только чтение). OOF-массивы,
веса и метрики — в отдельную папку, чтобы не затирать артефакты заказчика.

Запуск:
    python dxa_qc_work/scripts/train_gpu.py --epochs 30 --aux-mask
"""
from __future__ import annotations

import argparse
import os
import sys

import _common  # noqa: F401

from src import config as C        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--aux-mask", action="store_true")
    ap.add_argument("--backbone", default=C.BACKBONE)
    ap.add_argument("--batch", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--tag", default="v2")
    args = ap.parse_args()

    # --- перенаправление выходов в рабочую папку ---
    art = os.path.join(_common.OUT, "artifacts")
    os.makedirs(art, exist_ok=True)
    C.ARTIFACTS_DIR = art
    C.CACHE_DIR = os.path.join(art, "image_cache")
    C.BEST_WEIGHTS = os.path.join(art, f"best_model_{args.tag}.pt")
    C.FOLD_WEIGHTS_TMPL = os.path.join(art, f"fold_{args.tag}_{{k}}.pt")
    C.METRICS_JSON = os.path.join(art, f"metrics_{args.tag}.json")
    # манифест оставляем репозиторный (read-only), чтобы не пересобирать кэш
    manifest = C.MANIFEST_CSV

    sys.argv = ["train_v2", "--epochs", str(args.epochs), "--manifest", manifest,
                "--backbone", args.backbone, "--batch", str(args.batch),
                "--lr", str(args.lr), "--device", "cuda"]
    if args.aux_mask:
        sys.argv.append("--aux-mask")

    from src import train_v2
    print(f"[gpu] артефакты: {art}")
    print(f"[gpu] манифест:  {manifest}")
    print(f"[gpu] epochs={args.epochs} aux_mask={args.aux_mask} backbone={args.backbone}")
    train_v2.main()


if __name__ == "__main__":
    main()
