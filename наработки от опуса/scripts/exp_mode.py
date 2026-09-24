# -*- coding: utf-8 -*-
"""Корректное сравнение режимов порога на конфигурации v3 (один харнесс honest_eval).

honest_eval принимает mode: 'f1' (сетка, как в развёрнутом пайплайне) или 'prior'
(доля нарушений). Это apples-to-apples проверка решающего слоя.

Запуск:
    python dxa_qc_work/scripts/exp_mode.py --seeds 20
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

from exp_sweep import CriterionScores, honest_eval  # noqa: E402
from build_v3 import V3_FEATURES, V3_SOURCES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(_common.OUT, "exp_mode.json"))
    args = ap.parse_args()

    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    cs = CriterionScores(data, real)
    S = {n: cs.get(n, V3_FEATURES[n], V3_SOURCES[n]) for n in C.VIOLATIONS}

    res = {}
    for mode in ("f1", "prior"):
        r = honest_eval(data, real, S, n_seeds=args.seeds, mode=mode)
        res[mode] = r
        print(f"mode={mode:6} macro-F1={r['macro']:.3f} ± {r['macro_std']:.3f}  "
              f"BA={r['ba']:.3f} qF1={r['qf1']:.3f} AUC={r['auc']:.3f}")
        print("   ", {k: round(v, 3) for k, v in r["per_label"].items()})

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\nсохранено: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
