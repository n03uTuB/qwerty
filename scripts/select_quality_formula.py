# -*- coding: utf-8 -*-
"""Выбор формулы quality_class, максимизирующей метрики организатора.

Так как quality_class почти совпадает с ИЛИ(критериев) (246/249), проверяем:
  B) quality = ИЛИ(сработавших критериев);
  C) quality = ИЛИ(критериев) ИЛИ (CNN-качество области >= порога);
  D) quality = ИЛИ(критериев) ИЛИ (CNN-качество >= общего порога).
Для каждой — balanced accuracy / macro-F1 / ROC-AUC по quality_class.
Macro-F1 по 4 меткам от выбора quality_class не зависит.

    cd dxa_qc && python ../scripts/select_quality_formula.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dxa_qc"))

from src import config as C      # noqa: E402
from src import dataset as ds    # noqa: E402


def main() -> None:
    m = ds.load_manifest(C.MANIFEST_CSV)
    real = np.asarray(ds.real_mask(m))
    oof_q = np.load(os.path.join(C.ARTIFACTS_DIR, "oof_quality.npy"))
    oof_v = np.load(os.path.join(C.ARTIFACTS_DIR, "oof_violation.npy"))
    sp = os.path.join(C.ARTIFACTS_DIR, "stack_oof_violation.npy")
    if os.path.isfile(sp):
        sv = np.load(sp)
        if (~np.isnan(sv)).sum():
            oof_v = sv
    with open(os.path.join(C.ARTIFACTS_DIR, "thresholds.json"), encoding="utf-8") as f:
        th = json.load(f)
    vth = th["violations"]
    qbr = th.get("quality_by_region", {})
    qthr_glob = th.get("quality", {}).get("threshold", 0.5)

    df = m.reset_index(drop=True)
    n = len(df)
    fired_any = np.zeros(n, dtype=int)
    max_viol = np.full(n, np.nan)
    for i, region in enumerate(df["region"].values):
        for name in C.REGION_CRITERIA[region]:
            p = oof_v[i, C.VIOLATION_IDX[name]]
            if not np.isnan(p):
                max_viol[i] = p if np.isnan(max_viol[i]) else max(max_viol[i], p)
            if p >= vth.get(name, {}).get("threshold", 0.5):
                fired_any[i] = 1

    yq = df["quality"].values.astype(float)
    known = ~np.isnan(yq) & real
    yq_bin = np.nan_to_num(yq, nan=0.0).astype(int)[known]

    q_region_ok = np.zeros(n, dtype=int)
    for i, region in enumerate(df["region"].values):
        q_region_ok[i] = int(oof_q[i] >= qbr.get(region, {}).get("threshold", qthr_glob))
    q_glob_ok = (oof_q >= qthr_glob).astype(int)

    variants = {
        "A гейт (как сейчас)": (q_region_ok, oof_q),          # заглушка
        "B ИЛИ(критериев)": (fired_any, max_viol),
        "C ИЛИ(крит) | CNN-обл": (((fired_any + q_region_ok) > 0).astype(int),
                                  np.fmax(max_viol, oof_q)),
        "D ИЛИ(крит) | CNN-общ": (((fired_any + q_glob_ok) > 0).astype(int),
                                  np.fmax(max_viol, oof_q)),
    }
    print(f"{'вариант':26} {'bal.acc':>8} {'macroF1':>8} {'AUC':>7} {'доля':>7}")
    for name, (pred, prob) in variants.items():
        p = pred[known]
        pr = prob[known]
        ok = ~np.isnan(pr)
        auc = roc_auc_score(yq_bin[ok], pr[ok]) if len(np.unique(yq_bin[ok])) > 1 else float("nan")
        print(f"{name:26} {balanced_accuracy_score(yq_bin, p):8.3f} "
              f"{f1_score(yq_bin, p, average='macro', zero_division=0):8.3f} "
              f"{auc:7.3f} {p.mean():7.3f}")


if __name__ == "__main__":
    main()
