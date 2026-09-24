# -*- coding: utf-8 -*-
"""Диагностика: почему просел ROC-AUC quality_class после femur_positioning->geo.

quality_class (решение) = OR(сработавших критериев) и от выбора шкалы для
quality_prob не зависит. ROC-AUC же считается по quality_prob — максимуму
вероятностей критериев области. После перехода бедра на «geo» шкала вероятности
изменилась, и AUC просел. Здесь сравниваются варианты определения quality_prob
при неизменном решении:

  deployed — max по критериям области от БОЕВЫХ скоров (geo для бедра);
  cnn-max  — max по критериям области от сырых CNN-скоров (как было до итерации 3);
  head     — вероятность CNN-головы качества (region_idx).

Печатаются balanced accuracy / macro-F1 (от решения, не зависят от варианта) и
ROC-AUC по каждому варианту.

Запуск:
    cd dxa_qc && python ../scripts/diag_quality_prob.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dxa_qc"))

from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402
from src import stack as st        # noqa: E402
from src.train import pick_threshold  # noqa: E402

import eval_honest_organizer as E  # noqa: E402

N_SEEDS = 20


def run(data, real, S, sm, oof_q, seed):
    idx = np.where(real)[0]
    g = data["study_uid"].values[idx]
    strat = data["region_idx"].values[idx] if "region_idx" in data.columns else None
    if strat is None or len(np.unique(strat)) < 2:
        splits = GroupKFold(n_splits=5).split(idx, groups=g)
    else:
        splits = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                      random_state=seed).split(idx, strat, groups=g)
    fired = np.zeros((len(data), len(C.VIOLATIONS)), dtype=int)
    qp_dep = np.full(len(data), np.nan)
    qp_cnn = np.full(len(data), np.nan)
    qp_head = np.full(len(data), np.nan)
    for tr_i, va_i in splits:
        tr, va = idx[tr_i], idx[va_i]
        for j, name in enumerate(C.VIOLATIONS):
            key = "spine" if name.startswith("spine") else "femur"
            rmask = data["region"].str.contains(key).values
            y = data["viol_" + name].values.astype(float)
            p = S[sm[name]][:, j]
            trc = tr[rmask[tr] & ~np.isnan(y[tr]) & ~np.isnan(p[tr])]
            if len(trc) < 2 or len(np.unique(y[trc].astype(int))) < 2:
                continue
            thr, _ = pick_threshold(y[trc].astype(int), p[trc], C.THRESHOLD_MODE)
            va_ok = va[~np.isnan(p[va])]
            fired[va_ok[p[va_ok] >= thr], j] = 1
        for i in va:
            crit = C.REGION_CRITERIA[data["region"].values[i]]
            vd = [float(S[sm[n]][i, C.VIOLATION_IDX[n]]) for n in crit]
            vc = [float(S["cnn"][i, C.VIOLATION_IDX[n]]) for n in crit]
            vd = [v for v in vd if not np.isnan(v)]
            vc = [v for v in vc if not np.isnan(v)]
            if vd:
                qp_dep[i] = max(vd)
            if vc:
                qp_cnn[i] = max(vc)
            ri = C.REGION_TO_IDX[data["region"].values[i]]
            qp_head[i] = float(oof_q[i]) if not np.isnan(oof_q[i]) else np.nan

    yq = data["quality"].values.astype(float)
    kn = ~np.isnan(yq) & real
    yqb = np.nan_to_num(yq, nan=0.0).astype(int)[kn]
    pq = (fired.sum(axis=1) > 0).astype(int)[kn]
    out = dict(ba=balanced_accuracy_score(yqb, pq),
               qf1=f1_score(yqb, pq, average="macro", zero_division=0))
    for tag, arr in (("deployed", qp_dep), ("cnn-max", qp_cnn), ("head", qp_head)):
        p = arr[kn]
        ok = ~np.isnan(p)
        out["auc_" + tag] = (roc_auc_score(yqb[ok], p[ok])
                             if len(np.unique(yqb[ok])) > 1 else float("nan"))
    return out


def main() -> None:
    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    S = E.build_scores(data, real)
    sm = {c: st.source_of(c) for c in C.VIOLATIONS}
    oof_q = np.load(os.path.join(C.ARTIFACTS_DIR, "oof_quality.npy"))
    print(f"источники: {sm}\nрежим: {C.THRESHOLD_MODE}\n")
    res = [run(data, real, S, sm, oof_q, s) for s in range(N_SEEDS)]
    for key in ("ba", "qf1", "auc_deployed", "auc_cnn-max", "auc_head"):
        v = np.array([r[key] for r in res])
        print(f"  {key:14} {v.mean():.3f} ± {v.std():.3f}")


if __name__ == "__main__":
    main()
