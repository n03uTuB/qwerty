# -*- coding: utf-8 -*-
"""Честный выбор режима порога на каждый критерий (внутри train-фолда).

Оракульный выбор «лучшего режима на метку» даёт оптимистичную оценку. Здесь режим
для каждого критерия выбирается по ВНУТРЕННЕЙ GroupKFold на train-части: берём
режим с максимальным F1 критерия на внутренней валидации, применяем к внешней.
Сравниваем с фиксированным f1 (развёрнутый) и prior.

Запуск:
    python dxa_qc_work/scripts/exp_per_label_mode.py --seeds 20
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import _common  # noqa: F401

from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402
from src.train import pick_threshold  # noqa: E402

from exp_sweep import CriterionScores, ORG_LABELS  # noqa: E402
from build_v3 import V3_FEATURES, V3_SOURCES  # noqa: E402
from exp_thresh import THR  # noqa: E402

MODES = ["f1", "prior", "prior1.2", "prior0.8", "youden"]


def choose_mode(p, y, g, modes):
    """Режим с лучшим F1 критерия на внутренней GroupKFold."""
    scores = {}
    for m in modes:
        f1s = []
        for tr, va in GroupKFold(n_splits=5).split(p, y, groups=g):
            if len(np.unique(y[tr])) < 2:
                continue
            thr = THR[m](y[tr], p[tr])
            f1s.append(f1_score(y[va], (p[va] >= thr).astype(int), zero_division=0))
        scores[m] = float(np.mean(f1s)) if f1s else 0.0
    return max(scores, key=scores.get)


def run(data, real, S, n_seeds, per_label):
    idx = np.where(real)[0]
    g = data["study_uid"].values[idx]
    strat = data["region_idx"].values[idx]
    macros, per = [], {k: [] for k in ORG_LABELS}
    for seed in range(n_seeds):
        splits = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                      random_state=seed).split(idx, strat, groups=g)
        fired = np.zeros((len(data), len(C.VIOLATIONS)), dtype=int)
        for tr_i, va_i in splits:
            tr, va = idx[tr_i], idx[va_i]
            for j, name in enumerate(C.VIOLATIONS):
                reg = "spine" if name.startswith("spine") else "femur"
                rmask = data["region"].str.contains(reg).values
                y = data["viol_" + name].values.astype(float)
                p = S[name]
                trc = tr[rmask[tr] & ~np.isnan(y[tr]) & ~np.isnan(p[tr])]
                if len(trc) < 10 or len(np.unique(y[trc].astype(int))) < 2:
                    continue
                if per_label:
                    mode = choose_mode(p[trc], y[trc].astype(int),
                                       data["study_uid"].values[trc], MODES)
                else:
                    mode = "f1"
                thr = THR[mode](y[trc].astype(int), p[trc])
                va_ok = va[~np.isnan(p[va])]
                fired[va_ok[p[va_ok] >= thr], j] = 1
        fs = []
        for lab, crits in ORG_LABELS.items():
            gt = np.zeros(len(data), dtype=int)
            pr = np.zeros(len(data), dtype=int)
            for name in crits:
                gt |= np.nan_to_num(data["viol_" + name].values.astype(float), nan=0.0).astype(int)
                pr |= fired[:, C.VIOLATION_IDX[name]]
            f = f1_score(gt[idx], pr[idx], zero_division=0)
            per[lab].append(f)
            fs.append(f)
        macros.append(float(np.mean(fs)))
    return (float(np.mean(macros)), float(np.std(macros)),
            {k: float(np.mean(v)) for k, v in per.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(_common.OUT, "exp_per_label_mode.json"))
    args = ap.parse_args()

    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    cs = CriterionScores(data, real)
    S = {n: cs.get(n, V3_FEATURES[n], V3_SOURCES[n]) for n in C.VIOLATIONS}

    res = {}
    for tag, per in (("f1 (развёрнутый)", False), ("выбор режима на критерий", True)):
        m, s, pl = run(data, real, S, args.seeds, per)
        res[tag] = {"macro": m, "std": s, "per_label": pl}
        print(f"{tag:26} macro-F1={m:.3f} ± {s:.3f}  "
              + " ".join(f"{k}={v:.3f}" for k, v in pl.items()))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\nсохранено: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
