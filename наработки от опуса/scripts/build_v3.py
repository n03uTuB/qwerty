# -*- coding: utf-8 -*-
"""Сборка улучшенной конфигурации v3 (без изменения mogaem).

Что делает:
  1. считает OOF-скоры критериев для улучшенной конфигурации;
  2. честно оценивает macro-F1 организатора (20 разбиений) и печатает по меткам;
  3. обучает финальные стекеры (для инференса) и сохраняет их;
  4. калибрует пороги по OOF и сохраняет thresholds.

Всё пишется в dxa_qc_work/out/v3/, исходный репозиторий не изменяется.

Улучшения относительно базы (подтверждены парно, 20 разбиений):
  * spine_axis: в гибрид добавлен признак spine_midline_residual (+0.049, 20/20);
  * spine_artifacts: источник fused вместо cnn (+0.007, 15/20).
Вместе: macro-F1 0.507 -> 0.563 (+0.056, 19/20).

Запуск:
    python dxa_qc_work/scripts/build_v3.py
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np

import _common  # noqa: F401

import joblib
from sklearn.model_selection import GroupKFold

from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402
from src import stack as st        # noqa: E402
from src.train import pick_threshold  # noqa: E402

from exp_sweep import CriterionScores, honest_eval  # noqa: E402

V3_DIR = os.path.join(_common.OUT, "v3")

V3_FEATURES = {
    "spine_positioning": ["spine_iliac_signal", "spine_bottom_cut"],
    "spine_axis": ["spine_midline_angle", "spine_midline_residual"],
    "spine_artifacts": ["spine_ribs_signal", "spine_vertebra_peaks"],
    "femur_positioning": ["femur_margin_min_cm", "femur_width_cm"],
    "femur_roi": ["femur_height_cm", "femur_bone_ratio", "femur_margin_min_cm"],
}
V3_SOURCES = {
    "spine_positioning": "cnn",
    "spine_axis": "fused",
    "spine_artifacts": "fused",
    "femur_positioning": "geo",
    "femur_roi": "cnn",
}


def fit_final_stackers(data, real):
    """Финальные стекеры на всех реальных данных (для инференса)."""
    oof_v = np.load(os.path.join(C.ARTIFACTS_DIR, "oof_violation.npy"))
    stackers = {}
    for j, name in enumerate(C.VIOLATIONS):
        src = V3_SOURCES[name]
        cols = V3_FEATURES[name]
        if src not in ("fused", "geo") or not cols:
            continue
        yv = data["viol_" + name].values.astype(float)
        val = ~np.isnan(yv) & ~np.isnan(oof_v[:, j]) & real
        if val.sum() == 0 or len(np.unique(yv[val])) < 2:
            continue
        Xv = data[cols].fillna(0).values
        use_cnn = src == "fused"
        base = np.column_stack([oof_v[val, j], Xv[val]]) if use_cnn else Xv[val]
        clf = st._make_clf().fit(base, yv[val].astype(int))
        stackers[name] = dict(clf=clf, features=cols, use_cnn=use_cnn)
    return stackers


def calibrate(data, real, S):
    """Пороги по OOF для улучшенной конфигурации (режим config.THRESHOLD_MODE)."""
    vth = {}
    for name in C.VIOLATIONS:
        y = data["viol_" + name].values.astype(float)
        p = S[name]
        m = ~np.isnan(y) & ~np.isnan(p) & real
        if m.sum() and len(np.unique(y[m])) > 1:
            thr, f1 = pick_threshold(y[m].astype(int), p[m], C.THRESHOLD_MODE)
            vth[name] = dict(threshold=float(thr), f1=float(f1),
                             n=int(m.sum()), pos=int((y[m] > 0.5).sum()))
        else:
            vth[name] = dict(threshold=0.5, f1=None, n=int(m.sum()))
    return {"violations": vth, "mode": C.THRESHOLD_MODE,
            "features": V3_FEATURES, "sources": V3_SOURCES}


def main():
    os.makedirs(V3_DIR, exist_ok=True)
    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    print(f"строк {len(data)} реальных {int(real.sum())}")

    cs = CriterionScores(data, real)
    S = {n: cs.get(n, V3_FEATURES[n], V3_SOURCES[n]) for n in C.VIOLATIONS}

    res = honest_eval(data, real, S, n_seeds=20)
    print("\n=== УЛУЧШЕННАЯ КОНФИГУРАЦИЯ v3 (честно, 20 разбиений) ===")
    print(f"  macro-F1 (4 метки)  {res['macro']:.3f} ± {res['macro_std']:.3f}")
    for k, v in res["per_label"].items():
        print(f"    {k:10} {v:.3f}")
    print(f"  quality BA {res['ba']:.3f}   macro-F1 {res['qf1']:.3f}   ROC-AUC {res['auc']:.3f}")

    # сохраняем OOF-скоры улучшенной конфигурации
    np.save(os.path.join(V3_DIR, "stack_oof_violation_v3.npy"),
            np.column_stack([S[n] for n in C.VIOLATIONS]))

    stackers = fit_final_stackers(data, real)
    joblib.dump(stackers, os.path.join(V3_DIR, "stackers_v3.joblib"))
    thr = calibrate(data, real, S)
    with open(os.path.join(V3_DIR, "thresholds_v3.json"), "w", encoding="utf-8") as f:
        json.dump(thr, f, ensure_ascii=False, indent=2)
    with open(os.path.join(V3_DIR, "metrics_v3.json"), "w", encoding="utf-8") as f:
        json.dump({"honest": res, "config": {"features": V3_FEATURES, "sources": V3_SOURCES}},
                  f, ensure_ascii=False, indent=2)

    # копии входных OOF, чтобы бандл был самодостаточным
    for fn in ("oof_violation.npy", "oof_quality.npy", "manifest.csv", "features.csv"):
        src = os.path.join(C.ARTIFACTS_DIR, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(V3_DIR, fn))

    print(f"\nбандл сохранён: {V3_DIR}")
    print("  stack_oof_violation_v3.npy, stackers_v3.joblib, thresholds_v3.json, metrics_v3.json")


if __name__ == "__main__":
    main()
