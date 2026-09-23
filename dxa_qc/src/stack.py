# -*- coding: utf-8 -*-
"""Стеккинг (гибрид): нейросетевые вероятности + геометрические признаки.

Идея: часть критериев — измеримые геометрические величины (ось позвоночника,
наклон бедра). Логистическая регрессия поверх OOF-вероятностей CNN и
геометрических признаков даёт калиброванную гибридную оценку.

Обучение стекера — с кросс-валидацией по study_uid (без утечки).
Артефакт: artifacts/stacker.joblib + artifacts/stack_metrics.json.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from . import dataset as ds
from . import features as feat_lib
from .train import best_threshold, bootstrap_ci

STACKER_PATH = os.path.join(C.ARTIFACTS_DIR, "stacker.joblib")
STACK_METRICS = os.path.join(C.ARTIFACTS_DIR, "stack_metrics.json")

# Какие геометрические признаки использовать для каждого критерия.
# Наборы перенесены из честного контура dxa_real и подобраны по приросту
# ROC-AUC на OOF-CV. Ключевые принципы: минимум признаков (на 6–36 позитивах
# широкая модель переобучается) и семантическое соответствие критерию ТЗ.
CRITERION_FEATURES = {
    # укладка позвоночника: видимость гребней подвздошных костей снизу
    "spine_positioning": ["spine_iliac_signal", "spine_bottom_cut"],
    # угол по средней линии столба — прямое измерение критерия ТЗ (допуск 5°)
    "spine_axis": ["spine_midline_angle"],
    # посторонние предметы: рёбра/Th12 сверху и число тел позвонков
    "spine_artifacts": ["spine_ribs_signal", "spine_vertebra_peaks"],
    # ротация бедра: отступ поля сканирования + ширина кадра (набор O).
    # Победитель перебора в режиме prior (дефолт пайплайна): macro-F1 0.500 -> 0.517;
    # выигрывает и в nested (0.404 -> 0.410). Лучший глобальный AUC даёт другой
    # набор (L, 0.625), но в режиме порога по доле нарушений важна точность верхушки
    # списка, и там O сильнее.
    "femur_positioning": ["femur_margin_min_cm", "femur_width_cm"],
    # область интереса: высота кадра и доля кости (отступы поля сканирования)
    "femur_roi": ["femur_height_cm", "femur_bone_ratio", "femur_margin_min_cm"],
}
QUALITY_FEATURES = []

# Позволяем переопределить наборы через config (для A/B без правки кода).
CRITERION_FEATURES.update(getattr(C, "CRITERION_FEATURES_OVERRIDE", {}) or {})


def _make_clf():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   C=1.0)),
    ])


def _cv_fuse(X, cnn_prob, y, groups, n_splits=5):
    """Кросс-валидированный гибридный скор: [cnn_prob | features] -> LR."""
    n = len(y)
    oof = np.full(n, np.nan)
    gkf = GroupKFold(n_splits=n_splits)
    Xf = np.column_stack([cnn_prob, X])
    for tr, va in gkf.split(Xf, groups=groups):
        clf = _make_clf()
        clf.fit(Xf[tr], y[tr])
        oof[va] = clf.predict_proba(Xf[va])[:, 1]
    return oof


def _cv_cnn_only(cnn_prob, y, groups, n_splits=5):
    Xf = cnn_prob.reshape(-1, 1)
    n = len(y)
    oof = np.full(n, np.nan)
    gkf = GroupKFold(n_splits=n_splits)
    for tr, va in gkf.split(Xf, groups=groups):
        clf = _make_clf()
        clf.fit(Xf[tr], y[tr])
        oof[va] = clf.predict_proba(Xf[va])[:, 1]
    return oof


def _score(y, p):
    if len(np.unique(y)) < 2:
        return dict(n=int(len(y)), f1=None, auc=None)
    thr, f1 = best_threshold(y, p)
    return dict(n=int(len(y)), pos=int((y > 0.5).sum()),
                f1=float(f1), auc=float(roc_auc_score(y, p)),
                thr=float(thr),
                f1_ci=bootstrap_ci(y, p, "f1", thr=thr),
                auc_ci=bootstrap_ci(y, p, "auc"))


def main():
    manifest = ds.load_manifest(C.MANIFEST_CSV)
    oof_q = np.load(os.path.join(C.ARTIFACTS_DIR, "oof_quality.npy"))
    oof_v = np.load(os.path.join(C.ARTIFACTS_DIR, "oof_violation.npy"))

    data = feat_lib.load_features(manifest)
    # стекер и его метрики — только по реальным снимкам (синтетика нужна лишь CNN).
    # Строки не выбрасываем, иначе разъедется индексация с OOF-массивами и манифестом.
    real = ds.real_mask(data)
    if not real.all():
        print(f"[stack] синтетических строк исключено из оценки: {int((~real).sum())}")
    groups = data["study_uid"].values
    results = {"quality": {}, "violations": {}, "comparison": {}}

    n = len(data)
    stack_q = np.full(n, np.nan)
    stack_v = np.full((n, len(C.VIOLATIONS)), np.nan)

    # --- качество ---
    y = data["quality"].values.astype(float)
    valid = ~np.isnan(y) & ~np.isnan(oof_q) & real
    X = data[QUALITY_FEATURES].fillna(0).values if QUALITY_FEATURES else \
        np.zeros((n, 0), dtype=float)
    idx = np.where(valid)[0]
    fused = _cv_fuse(X[idx], oof_q[idx], y[idx].astype(int), groups[idx])
    cnn_only = _cv_cnn_only(oof_q[idx], y[idx].astype(int), groups[idx])
    stack_q[idx] = fused
    results["quality"] = _score(y[idx], fused)
    results["comparison"]["quality_cnn_only"] = _score(y[idx], cnn_only)
    results["comparison"]["quality_fused"] = results["quality"]

    # --- нарушения ---
    for j, name in enumerate(C.VIOLATIONS):
        yv = data["viol_" + name].values.astype(float)
        val = ~np.isnan(yv) & ~np.isnan(oof_v[:, j]) & real
        if val.sum() == 0 or len(np.unique(yv[val])) < 2:
            results["violations"][name] = dict(n=int(val.sum()), f1=None, auc=None)
            continue
        idv = np.where(val)[0]
        cols = CRITERION_FEATURES.get(name, QUALITY_FEATURES)
        Xv = data[cols].fillna(0).values if cols else np.zeros((n, 0), dtype=float)
        fused_v = _cv_fuse(Xv[idv], oof_v[idv, j], yv[idv].astype(int), groups[idv])
        cnn_v = _cv_cnn_only(oof_v[idv, j], yv[idv].astype(int), groups[idv])
        stack_v[idv, j] = fused_v
        results["violations"][name] = _score(yv[idv], fused_v)
        results["comparison"][name + "_cnn_only"] = _score(yv[idv], cnn_v)
        results["comparison"][name + "_fused"] = results["violations"][name]

    # --- сохранить fused OOF-вероятности нарушений (для калибровки порогов) ---
    # Качество оставляем на «сырой» CNN-шкале: стекер по одному признаку
    # (вероятность CNN) ничего не добавляет и слегка ухудшает ранжирование.
    np.save(os.path.join(C.ARTIFACTS_DIR, "stack_oof_violation.npy"), stack_v)

    # --- финальные стекеры на всех данных (для инференса) ---
    try:
        import joblib

        stackers = {}
        for j, name in enumerate(C.VIOLATIONS):
            yv = data["viol_" + name].values.astype(float)
            val = ~np.isnan(yv) & ~np.isnan(oof_v[:, j]) & real
            if val.sum() and len(np.unique(yv[val])) > 1:
                cols = CRITERION_FEATURES.get(name, QUALITY_FEATURES)
                Xv = data[cols].fillna(0).values if cols else \
                    np.zeros((n, 0), dtype=float)
                clf = _make_clf()
                clf.fit(np.column_stack([oof_v[val, j], Xv[val]]), yv[val].astype(int))
                stackers[name] = dict(clf=clf, features=cols)
        joblib.dump(stackers, STACKER_PATH)
        print("[stack] стекеры сохранены:", STACKER_PATH)
    except Exception as e:
        print("[stack] не удалось сохранить стекеры:", e)

    with open(STACK_METRICS, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=float)

    print("\n=== Гибрид (CNN + геометрия) vs только CNN ===")
    c = results["comparison"]
    print("quality:  CNN F1=%.3f AUC=%.3f  ->  fused F1=%.3f AUC=%.3f" % (
        c["quality_cnn_only"]["f1"], c["quality_cnn_only"]["auc"],
        c["quality_fused"]["f1"], c["quality_fused"]["auc"]))
    for name in C.VIOLATIONS:
        a = c.get(name + "_cnn_only", {})
        b = c.get(name + "_fused", {})
        if a.get("f1") is None or b.get("f1") is None:
            print("%-20s n/a" % name)
            continue
        print("%-20s CNN F1=%.3f AUC=%.3f  ->  fused F1=%.3f AUC=%.3f" % (
            name, a["f1"], a["auc"], b["f1"], b["auc"]))
    print("\n[stack] метрики:", STACK_METRICS)


if __name__ == "__main__":
    main()
