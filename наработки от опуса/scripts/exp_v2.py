# -*- coding: utf-8 -*-
"""Базовая конфигурация пайплайна: воспроизведение честной macro-F1 (контроль).

Служит точкой отсчёта: если база здесь не 0.507, значит изменились данные/артефакты.
Читает mogaem, пишет в dxa_qc_work/out.

Запуск:
    python dxa_qc_work/scripts/exp_v2.py --seeds 20
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import _common  # noqa: F401

from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402

from exp_sweep import BASE_FEATURES, BASE_SOURCES, CriterionScores, honest_eval, ORG_LABELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(_common.OUT, "exp_baseline.json"))
    args = ap.parse_args()

    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    print(f"строк {len(data)} реальных {int(real.sum())} seeds={args.seeds}")

    cs = CriterionScores(data, real)
    S = {n: cs.get(n, BASE_FEATURES[n], BASE_SOURCES[n]) for n in C.VIOLATIONS}
    res = honest_eval(data, real, S, n_seeds=args.seeds)

    print(f"\nБАЗА (развёрнутая конфигурация)")
    print(f"  macro-F1 (4 метки)  {res['macro']:.3f} ± {res['macro_std']:.3f}")
    for k, v in res["per_label"].items():
        print(f"    {k:10} {v:.3f}")
    print(f"  quality BA {res['ba']:.3f}   macro-F1 {res['qf1']:.3f}   ROC-AUC {res['auc']:.3f}")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"seeds": args.seeds, "baseline": res}, fh, ensure_ascii=False, indent=2)
    print(f"\nсохранено: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
