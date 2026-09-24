# -*- coding: utf-8 -*-
"""Перебор признаков/источников по критериям (честная macro-F1 организатора).

Читает данные и код из репозитория mogaem, пишет результаты в dxa_qc_work/out.
Протокол: только реальные снимки, 20 разбиений StratifiedGroupKFold по study_uid,
порог по train-части фолда, macro-F1 по 4 меткам (укладка, ось, предметы, ROI).

Запуск:
    python dxa_qc_work/scripts/exp_sweep.py --seeds 20
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import _common  # noqa: F401  (настраивает sys.path на репозиторий)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402
from src import stack as st        # noqa: E402
from src.train import pick_threshold  # noqa: E402

ORG_LABELS = {
    "укладка": ["spine_positioning", "femur_positioning"],
    "ось": ["spine_axis"],
    "предметы": ["spine_artifacts"],
    "ROI": ["femur_roi"],
}
N_SEEDS = 20

BASE_FEATURES = {
    "spine_positioning": ["spine_iliac_signal", "spine_bottom_cut"],
    "spine_axis": ["spine_midline_angle"],
    "spine_artifacts": ["spine_ribs_signal", "spine_vertebra_peaks"],
    "femur_positioning": ["femur_margin_min_cm", "femur_width_cm"],
    "femur_roi": ["femur_height_cm", "femur_bone_ratio", "femur_margin_min_cm"],
}
BASE_SOURCES = {c: st.source_of(c) for c in C.VIOLATIONS}


def _clf():
    return Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))])


def _cv(X, y, groups, cnn=None):
    oof = np.full(len(y), np.nan)
    if len(np.unique(y)) < 2:
        return oof
    Xf = X if cnn is None else np.column_stack([cnn, X])
    for tr, va in GroupKFold(n_splits=5).split(Xf, y, groups=groups):
        if len(np.unique(y[tr])) < 2:
            continue
        oof[va] = _clf().fit(Xf[tr], y[tr]).predict_proba(Xf[va])[:, 1]
    return oof


class CriterionScores:
    """Кэш OOF-скоров по критерию: (name, cols, source) -> вектор."""

    def __init__(self, data, real):
        self.data = data
        self.real = real
        oof_path = os.environ.get("DXA_OOF_V", os.path.join(C.ARTIFACTS_DIR, "oof_violation.npy"))
        self.oof_cnn = np.load(oof_path)
        self.cache = {}

    def get(self, name, cols, source):
        key = (name, tuple(cols), source)
        if key in self.cache:
            return self.cache[key]
        j = C.VIOLATION_IDX[name]
        reg = "spine" if name.startswith("spine") else "femur"
        rmask = self.data["region"].str.contains(reg).values & self.real
        y = self.data["viol_" + name].values.astype(float)
        known = ~np.isnan(y) & rmask
        out = np.full(len(self.data), np.nan)
        if source == "cnn":
            out[known] = self.oof_cnn[known, j]
        else:
            cols = [c for c in cols if c in self.data.columns]
            X = self.data.loc[known, cols].fillna(0).values if cols else np.zeros((known.sum(), 1))
            yk = y[known].astype(int)
            gk = self.data.loc[known, "study_uid"].values
            out[known] = _cv(X, yk, gk, cnn=(self.oof_cnn[known, j] if source == "fused" else None))
        self.cache[key] = out
        return out


def honest_eval(data, real, S_cols, n_seeds=N_SEEDS, mode=None):
    mode = mode or C.THRESHOLD_MODE
    idx = np.where(real)[0]
    g = data["study_uid"].values[idx]
    strat = data["region_idx"].values[idx]
    macros, per_label = [], {k: [] for k in ORG_LABELS}
    bas, qf1s, aucs = [], [], []
    for seed in range(n_seeds):
        splits = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                      random_state=seed).split(idx, strat, groups=g)
        fired = np.zeros((len(data), len(C.VIOLATIONS)), dtype=int)
        qprob = np.full(len(data), np.nan)
        for tr_i, va_i in splits:
            tr, va = idx[tr_i], idx[va_i]
            for j, name in enumerate(C.VIOLATIONS):
                reg = "spine" if name.startswith("spine") else "femur"
                rmask = data["region"].str.contains(reg).values
                y = data["viol_" + name].values.astype(float)
                p = S_cols[name]
                trc = tr[rmask[tr] & ~np.isnan(y[tr]) & ~np.isnan(p[tr])]
                if len(trc) < 2 or len(np.unique(y[trc].astype(int))) < 2:
                    continue
                thr, _ = pick_threshold(y[trc].astype(int), p[trc], mode)
                va_ok = va[~np.isnan(p[va])]
                fired[va_ok[p[va_ok] >= thr], j] = 1
            for i in va:
                crit = C.REGION_CRITERIA[data["region"].values[i]]
                vals = [S_cols[n][i] for n in crit if not np.isnan(S_cols[n][i])]
                if vals:
                    qprob[i] = max(vals)
        fs = []
        for lab, crits in ORG_LABELS.items():
            gt = np.zeros(len(data), dtype=int)
            pr = np.zeros(len(data), dtype=int)
            for name in crits:
                gt |= np.nan_to_num(data["viol_" + name].values.astype(float), nan=0.0).astype(int)
                pr |= fired[:, C.VIOLATION_IDX[name]]
            f = f1_score(gt[idx], pr[idx], zero_division=0)
            per_label[lab].append(f)
            fs.append(f)
        macros.append(float(np.mean(fs)))
        yq = data["quality"].values.astype(float)
        kn = ~np.isnan(yq) & real
        yqb = np.nan_to_num(yq, nan=0.0).astype(int)[kn]
        pq = (fired.sum(axis=1) > 0).astype(int)[kn]
        bas.append(balanced_accuracy_score(yqb, pq))
        qf1s.append(f1_score(yqb, pq, average="macro", zero_division=0))
        prq = qprob[kn]
        ok = ~np.isnan(prq)
        aucs.append(roc_auc_score(yqb[ok], prq[ok]) if len(np.unique(yqb[ok])) > 1 else np.nan)
    return dict(macro=float(np.mean(macros)), macro_std=float(np.std(macros)),
                per_label={k: float(np.mean(v)) for k, v in per_label.items()},
                ba=float(np.nanmean(bas)), qf1=float(np.nanmean(qf1s)),
                auc=float(np.nanmean(aucs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--out", default=os.path.join(_common.OUT, "exp_sweep.json"))
    args = ap.parse_args()
    n_seeds = args.seeds

    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    print(f"строк {len(data)} реальных {int(real.sum())}  seeds={n_seeds}\n")

    cs = CriterionScores(data, real)
    S = {n: cs.get(n, BASE_FEATURES[n], BASE_SOURCES[n]) for n in C.VIOLATIONS}
    base = honest_eval(data, real, S, n_seeds=n_seeds)
    print(f"БАЗА  macro-F1={base['macro']:.3f}  BA={base['ba']:.3f} AUC={base['auc']:.3f}")
    print("  ", {k: round(v, 3) for k, v in base["per_label"].items()})

    trials = {
        "ось: +residual(geo)": ("spine_axis", ["spine_midline_angle", "spine_midline_residual"], "geo"),
        "ось: +axis(geo)": ("spine_axis", ["spine_midline_angle", "spine_axis_angle"], "geo"),
        "ось: +iliac(geo)": ("spine_axis", ["spine_midline_angle", "spine_iliac_signal"], "geo"),
        "ось: 3feat(geo)": ("spine_axis", ["spine_midline_angle", "spine_midline_residual", "spine_axis_angle"], "geo"),
        "ось: midline(fused)": ("spine_axis", ["spine_midline_angle"], "fused"),
        "ось: +res(fused)": ("spine_axis", ["spine_midline_angle", "spine_midline_residual"], "fused"),
        "пред: foreign(fused)": ("spine_artifacts", ["foreign_count", "foreign_area", "foreign_max_area",
                                                     "foreign_compactness", "foreign_top"], "fused"),
        "пред: ribs+foreign(fused)": ("spine_artifacts", ["spine_ribs_signal", "spine_vertebra_peaks",
                                                          "foreign_area", "foreign_count"], "fused"),
        "пред: geo": ("spine_artifacts", ["spine_ribs_signal", "spine_vertebra_peaks",
                                          "foreign_area", "foreign_count"], "geo"),
        "укладка: +margin": ("spine_positioning", ["spine_iliac_signal", "spine_bottom_cut",
                                                   "spine_margin_bottom_cm"], "fused"),
        "бедро: troch(geo)": ("femur_positioning", ["femur_trochanter_bulge", "femur_troch_area_mm2",
                                                    "femur_neck_width_mm"], "geo"),
        "бедро: troch(fused)": ("femur_positioning", ["femur_trochanter_bulge", "femur_troch_area_mm2",
                                                      "femur_neck_width_mm"], "fused"),
        "бедро: margin+troch(geo)": ("femur_positioning", ["femur_margin_min_cm", "femur_width_cm",
                                                           "femur_trochanter_bulge",
                                                           "femur_troch_area_mm2"], "geo"),
    }

    results = {"БАЗА": base}
    for title, (crit, cols, src) in trials.items():
        Sx = dict(S)
        Sx[crit] = cs.get(crit, cols, src)
        res = honest_eval(data, real, Sx, n_seeds=n_seeds)
        results[title] = res
        d = res["macro"] - base["macro"]
        org = next(k for k, v in ORG_LABELS.items() if crit in v)
        print(f"{title:30} macro-F1={res['macro']:.3f} ({d:+.3f})  | {org}: {res['per_label'][org]:.3f}")

    print("\n--- итог, отсортировано по macro-F1 ---")
    for title, res in sorted(results.items(), key=lambda kv: -kv[1]["macro"]):
        print(f"{title:30} {res['macro']:.3f}  BA={res['ba']:.3f} AUC={res['auc']:.3f}  "
              + " ".join(f"{k}={v:.3f}" for k, v in res["per_label"].items()))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"seeds": n_seeds, "results": results}, fh, ensure_ascii=False, indent=2)
    print(f"\nсохранено: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
