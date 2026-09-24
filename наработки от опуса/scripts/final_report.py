# -*- coding: utf-8 -*-
"""Итоговое сравнение: база vs v3 (AUC по критериям + честная macro-F1 по меткам).

Печатает таблицы для отчёта и сохраняет out/final_report.json.

Запуск:
    python dxa_qc_work/scripts/final_report.py --seeds 20
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

from exp_sweep import BASE_FEATURES, BASE_SOURCES, CriterionScores, honest_eval, ORG_LABELS  # noqa: E402
from build_v3 import V3_FEATURES, V3_SOURCES  # noqa: E402


def per_criterion_auc(data, real, S):
    out = {}
    for name in C.VIOLATIONS:
        y = data["viol_" + name].values.astype(float)
        p = S[name]
        m = ~np.isnan(y) & ~np.isnan(p) & real
        out[name] = (float(roc_auc_score(y[m].astype(int), p[m])) if len(np.unique(y[m])) > 1
                     else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(_common.OUT, "final_report.json"))
    args = ap.parse_args()

    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    cs = CriterionScores(data, real)

    base_S = {n: cs.get(n, BASE_FEATURES[n], BASE_SOURCES[n]) for n in C.VIOLATIONS}
    v3_S = {n: cs.get(n, V3_FEATURES[n], V3_SOURCES[n]) for n in C.VIOLATIONS}

    base = honest_eval(data, real, base_S, n_seeds=args.seeds)
    v3 = honest_eval(data, real, v3_S, n_seeds=args.seeds)
    base_auc = per_criterion_auc(data, real, base_S)
    v3_auc = per_criterion_auc(data, real, v3_S)

    print("=" * 66)
    print("ИТОГ: база vs v3 (честно, %d разбиений)" % args.seeds)
    print("=" * 66)
    print(f"{'метрика':22} {'база':>10} {'v3':>10} {'Δ':>9}")
    print(f"{'macro-F1 (4 метки)':22} {base['macro']:>10.3f} {v3['macro']:>10.3f} "
          f"{v3['macro']-base['macro']:>+9.3f}")
    for k in ORG_LABELS:
        print(f"{'  '+k:22} {base['per_label'][k]:>10.3f} {v3['per_label'][k]:>10.3f} "
              f"{v3['per_label'][k]-base['per_label'][k]:>+9.3f}")
    print(f"{'quality BA':22} {base['ba']:>10.3f} {v3['ba']:>10.3f} {v3['ba']-base['ba']:>+9.3f}")
    print(f"{'quality macro-F1':22} {base['qf1']:>10.3f} {v3['qf1']:>10.3f} "
          f"{v3['qf1']-base['qf1']:>+9.3f}")
    print(f"{'quality ROC-AUC':22} {base['auc']:>10.3f} {v3['auc']:>10.3f} "
          f"{v3['auc']-base['auc']:>+9.3f}")
    print("-" * 66)
    print(f"{'AUC по критерию':22} {'база':>10} {'v3':>10}")
    for name in C.VIOLATIONS:
        print(f"{'  '+name:22} {base_auc[name]:>10.3f} {v3_auc[name]:>10.3f}")

    rep = dict(seeds=args.seeds, base=base, v3=v3, base_auc=base_auc, v3_auc=v3_auc)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"\nсохранено: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
