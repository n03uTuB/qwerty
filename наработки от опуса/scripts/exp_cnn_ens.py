# -*- coding: utf-8 -*-
"""Ансамбль двух CNN: усреднение OOF (базовый + новый с GPU) в гибридной схеме v3.

Идея: даже если новый CNN по отдельности слабее, усреднение с базовым может
улучшить ранжирование за счёт разнообразия ошибок.

Запуск:
    python dxa_qc_work/scripts/exp_cnn_ens.py --new ..\\out\\artifacts\\oof_violation.npy --seeds 20
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import _common  # noqa: F401

from sklearn.metrics import roc_auc_score

from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402

from exp_sweep import CriterionScores, honest_eval  # noqa: E402
from build_v3 import V3_FEATURES, V3_SOURCES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(_common.OUT, "exp_cnn_ens.json"))
    args = ap.parse_args()

    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    base = np.load(os.path.join(C.ARTIFACTS_DIR, "oof_violation.npy"))
    new = np.load(args.new)
    if new.shape != base.shape:
        print("!! формы не совпадают"); return

    ens = (base + new) / 2.0
    ens_path = os.path.join(_common.OUT, "oof_violation_ens.npy")
    np.save(ens_path, ens)

    def v3_with(oof, tag):
        os.environ["DXA_OOF_V"] = oof
        cs = CriterionScores(data, real)
        S = {n: cs.get(n, V3_FEATURES[n], V3_SOURCES[n]) for n in C.VIOLATIONS}
        r = honest_eval(data, real, S, n_seeds=args.seeds)
        print(f"{tag:12} macro-F1={r['macro']:.3f} ± {r['macro_std']:.3f}  BA={r['ba']:.3f} "
              f"AUC={r['auc']:.3f}  " + " ".join(f"{k}={v:.3f}" for k, v in r['per_label'].items()))
        return r

    res = {}
    res["base_cnn"] = v3_with(os.path.join(C.ARTIFACTS_DIR, "oof_violation.npy"), "базовый")
    res["new_cnn"] = v3_with(os.path.abspath(args.new), "новый GPU")
    res["ens_cnn"] = v3_with(ens_path, "ансамбль")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\nсохранено: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
