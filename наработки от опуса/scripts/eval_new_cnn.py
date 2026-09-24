# -*- coding: utf-8 -*-
"""Оценка нового CNN (после обучения на GPU): AUC по критериям + честная macro-F1.

Сравнивает базовый OOF (репозиторий) и новый OOF (например,
out/artifacts/oof_violation.npy от train_gpu.py).

Запуск:
    python dxa_qc_work/scripts/eval_new_cnn.py --oof ..\out\artifacts\oof_violation.npy --seeds 20
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

from exp_sweep import CriterionScores, honest_eval, ORG_LABELS  # noqa: E402
from build_v3 import V3_FEATURES, V3_SOURCES  # noqa: E402


def criterion_auc(oof, data, real):
    out = {}
    for j, name in enumerate(C.VIOLATIONS):
        y = data["viol_" + name].values.astype(float)
        reg = "spine" if name.startswith("spine") else "femur"
        m = ~np.isnan(y) & real & data["region"].str.contains(reg).values
        p = oof[m, j]
        ok = ~np.isnan(p)
        out[name] = float(roc_auc_score(y[m][ok].astype(int), p[ok])) if len(np.unique(y[m][ok])) > 1 else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(_common.OUT, "eval_new_cnn.json"))
    args = ap.parse_args()

    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))

    base_oof = np.load(os.path.join(C.ARTIFACTS_DIR, "oof_violation.npy"))
    new_oof = np.load(args.oof)
    print(f"OOF базовый: {base_oof.shape}  новый: {new_oof.shape}")
    if new_oof.shape != base_oof.shape:
        print("!! формы не совпадают — пропускаю честную оценку"); return

    ba, na = criterion_auc(base_oof, data, real), criterion_auc(new_oof, data, real)
    print(f"\n{'критерий':22} {'AUC база':>10} {'AUC новый':>10}")
    for name in C.VIOLATIONS:
        print(f"  {name:20} {ba[name]:>10.3f} {na[name]:>10.3f}")
    print(f"  {'среднее':20} {np.nanmean(list(ba.values())):>10.3f} {np.nanmean(list(na.values())):>10.3f}")

    os.environ["DXA_OOF_V"] = os.path.abspath(args.oof)
    cs = CriterionScores(data, real)
    S = {n: cs.get(n, V3_FEATURES[n], V3_SOURCES[n]) for n in C.VIOLATIONS}
    res = honest_eval(data, real, S, n_seeds=args.seeds)
    print(f"\nv3 с новым CNN: macro-F1={res['macro']:.3f} ± {res['macro_std']:.3f}  "
          f"BA={res['ba']:.3f} AUC={res['auc']:.3f}")
    print("  ", {k: round(v, 3) for k, v in res["per_label"].items()})

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"base_auc": ba, "new_auc": na, "v3_new_cnn": res,
                   "oof": os.path.abspath(args.oof)}, f, ensure_ascii=False, indent=2)
    print(f"\nсохранено: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
