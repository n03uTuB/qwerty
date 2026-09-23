# -*- coding: utf-8 -*-
"""Замер честных метрик на реальном наборе организатора (dxa_real).

Запуск из корня проекта:
    python scripts/measure_real.py --root "C:\\...\\_dsroot"
    python scripts/measure_real.py --root ... --synthetic --use-meta
    python scripts/measure_real.py --root ... --threshold prior

Печатает таблицу метрик по критериям ТЗ и итоговому качеству, сохраняет
JSON в out/metrics_real.json для сравнения «до/после».
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dxa_real import evaluate as ev          # noqa: E402
from dxa_real import synth_real              # noqa: E402
from dxa_real.data import load_dataset       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="каталог с разметка.xlsx и исследованиями")
    ap.add_argument("--synthetic", action="store_true", help="добавить синтетические нарушения")
    ap.add_argument("--per-type", type=int, default=40)
    ap.add_argument("--use-meta", action="store_true", help="признаки copies/n_images")
    ap.add_argument("--threshold", default="nested", choices=["nested", "prior"])
    ap.add_argument("--out", default="out/metrics_real.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    images = load_dataset(args.root)
    print(f"[data] изображений: {len(images)}  исследований: {len({i.study_uid for i in images})}")

    if args.synthetic:
        rng = np.random.default_rng(args.seed)
        extra = synth_real.build(images, rng, per_type=args.per_type)
        print(synth_real.summary(extra))
        images = images + extra

    table = ev.build_table(images)
    res = ev.evaluate(images, table, verbose=True,
                      use_meta=args.use_meta, threshold_mode=args.threshold)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = {
        "config": dict(synthetic=args.synthetic, per_type=args.per_type,
                       use_meta=args.use_meta, threshold=args.threshold, seed=args.seed),
        "macro_f1": res["macro_f1"],
        "scores": {k: dict(n=v.n, positives=v.positives, auc=v.auc, auc_ci=list(v.auc_ci),
                           ap=v.ap, f1=v.f1, sensitivity=v.sensitivity, specificity=v.specificity)
                   for k, v in res["scores"].items()},
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[out] {args.out}   время: {time.time()-t0:.1f}s")
    print(f"[macro-F1 по критериям] {res['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
