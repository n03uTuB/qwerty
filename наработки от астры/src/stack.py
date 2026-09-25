# -*- coding: utf-8 -*-
"""Стеккинг (гибрид): вероятности CNN + геометрические признаки.

Часть критериев ТЗ измерима геометрически (ось позвоночника, наклон бедра),
поэтому логистическая регрессия поверх OOF-вероятностей CNN и признаков даёт
калиброванную гибридную оценку. Обучение стекера — CV по study_uid (без утечки).

Источник скора по критерию задаёт config.CRITERION_SOURCES (конфигурации v3/v4):
  * ``cnn``   — чистая сеть (стекер не строится);
  * ``fused`` — гибрид [cnn_prob | features] -> LogReg;
  * ``geo``   — чистая геометрия [features] -> LogReg.

Суффикс источника меняет классификатор (см. :mod:`src.stack_clf`):
``_lda`` — LDA со shrinkage, ``_bag`` — balanced bagging (EasyEnsemble).
Например, ``fused_bag`` = [cnn_prob | features] + balanced bagging.

Артефакты: artifacts/stacker.joblib + artifacts/stack_metrics.json +
artifacts/stack_oof_violation.npy (для калибровки порогов).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from . import config as C
from . import features as feat_lib
from .data import load_manifest, real_mask
from .metrics import best_threshold, bootstrap_ci
from .stack_clf import BalancedBag, fit_clf, make_clf, parse_source  # noqa: F401

STACKER_PATH = C.STACKER_PATH
STACK_METRICS = C.STACK_METRICS

# Какие геометрические признаки использовать для каждого критерия.
# Наборы перенесены из честного контура dxa_real и подобраны по приросту
# ROC-AUC на OOF-CV: минимум признаков (на 6-36 позитивах широкая модель
# переобучается) и семантическое соответствие критерию ТЗ.
CRITERION_FEATURES = {
    # укладка позвоночника: видимость гребней подвздошных костей снизу
    "spine_positioning": ["spine_iliac_signal", "spine_bottom_cut"],
    # угол по средней линии столба — прямое измерение критерия ТЗ (допуск 5 град)
    "spine_axis": ["spine_midline_angle"],
    # посторонние предметы: рёбра/Th12 сверху и число тел позвонков
    "spine_artifacts": ["spine_ribs_signal", "spine_vertebra_peaks"],
    # ротация бедра: отступ поля сканирования + ширина кадра
    "femur_positioning": ["femur_margin_min_cm", "femur_width_cm"],
    # область интереса: высота кадра и доля кости (отступы поля сканирования)
    "femur_roi": ["femur_height_cm", "femur_bone_ratio", "femur_margin_min_cm"],
}
QUALITY_FEATURES = []

# Переопределение наборов через config (конфигурация v3, без правки кода).
# DXA_FEATURE_MODE=base отключает переопределение — для честной абляции
# «до/после» (сравнение базовых наборов и новых art_*/atlas_*).
_FEATURE_MODE = os.environ.get("DXA_FEATURE_MODE", "new").strip().lower()
_OVERRIDE = dict(getattr(C, "CRITERION_FEATURES_OVERRIDE", {}) or {})
if _FEATURE_MODE == "base":
    # абляция: оставляем улучшение «оси» (v3), убираем новые наборы
    # предметов/укладки -> падаем на наборы по умолчанию выше
    for _k in ("spine_artifacts", "spine_positioning", "femur_positioning"):
        _OVERRIDE.pop(_k, None)
CRITERION_FEATURES.update(_OVERRIDE)
CRITERION_SOURCES = dict(getattr(C, "CRITERION_SOURCES", {}) or {})
if _FEATURE_MODE == "base":
    # в базовом режиме укладка позвоночника — чистая CNN (как в v3),
    # а укладка бедра — чистая геометрия на базовом наборе признаков (как в v3);
    # источники v4 (geo_bag/fused_bag) тоже откатываются к варианту v3.
    CRITERION_SOURCES["spine_positioning"] = "cnn"
    CRITERION_SOURCES["femur_positioning"] = "geo"
    CRITERION_SOURCES["spine_artifacts"] = "fused"
    CRITERION_SOURCES["femur_roi"] = "cnn"


def source_of(name: str) -> str:
    """Источник скора для критерия (по умолчанию — гибрид)."""
    return CRITERION_SOURCES.get(name, "fused")


def _cv_fuse(X, cnn_prob, y, groups, n_splits=5, kind="logreg"):
    """Кросс-валидированный гибридный скор: [cnn_prob | features] -> clf."""
    n = len(y)
    oof = np.full(n, np.nan)
    Xf = np.column_stack([cnn_prob, X])
    for tr, va in GroupKFold(n_splits=n_splits).split(Xf, groups=groups):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = fit_clf(kind, Xf[tr], y[tr])
        oof[va] = clf.predict_proba(Xf[va])[:, 1]
    return oof


def _cv_cnn_only(cnn_prob, y, groups, n_splits=5):
    Xf = cnn_prob.reshape(-1, 1)
    n = len(y)
    oof = np.full(n, np.nan)
    for tr, va in GroupKFold(n_splits=n_splits).split(Xf, groups=groups):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_clf()
        clf.fit(Xf[tr], y[tr])
        oof[va] = clf.predict_proba(Xf[va])[:, 1]
    return oof


def _cv_geo_only(X, y, groups, n_splits=5, kind="logreg"):
    """Кросс-валидированный скор только по геометрии: [features] -> clf."""
    n = len(y)
    oof = np.full(n, np.nan)
    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups=groups):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = fit_clf(kind, X[tr], y[tr])
        oof[va] = clf.predict_proba(X[va])[:, 1]
    return oof


def _score(y, p) -> dict:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return dict(n=int(len(y)), f1=None, auc=None)
    thr, f1 = best_threshold(y, p)
    return dict(n=int(len(y)), pos=int((y > 0.5).sum()),
                f1=float(f1), auc=float(roc_auc_score(y, p)), thr=float(thr),
                f1_ci=bootstrap_ci(y, p, "f1", thr=thr),
                auc_ci=bootstrap_ci(y, p, "auc"))


def build_scores(data: pd.DataFrame, oof_q: np.ndarray, oof_v: np.ndarray):
    """Построить OOF-скоры гибрида по каждому критерию.

    Возвращает (stack_q, stack_v, results). ``stack_q`` — сырая CNN-шкала
    качества (стекер по одной вероятности ничего не добавляет).
    """
    real = real_mask(data)
    groups = data["study_uid"].values
    n = len(data)
    stack_q = np.full(n, np.nan)
    stack_v = np.full((n, len(C.VIOLATIONS)), np.nan)
    results = {"quality": {}, "violations": {}, "comparison": {}}

    # --- качество: оставляем сырую CNN-шкалу (см. пояснение выше) ---
    y = data["quality"].values.astype(float)
    valid = ~np.isnan(y) & ~np.isnan(oof_q) & real
    idx = np.where(valid)[0]
    stack_q[idx] = oof_q[idx]
    results["quality"] = _score(y[idx], oof_q[idx])
    cnn_q = _cv_cnn_only(oof_q[idx], y[idx].astype(int), groups[idx])
    results["comparison"]["quality_cnn_only"] = _score(y[idx], cnn_q)
    results["comparison"]["quality_fused"] = results["quality"]

    # --- нарушения ---
    for j, name in enumerate(C.VIOLATIONS):
        yv = data["viol_" + name].values.astype(float)
        val = ~np.isnan(yv) & ~np.isnan(oof_v[:, j]) & real
        if val.sum() == 0 or len(np.unique(yv[val])) < 2:
            # нет валидных меток — оставляем сырую CNN-шкалу (без NaN в артефакте)
            stack_v[:, j] = oof_v[:, j]
            results["violations"][name] = dict(n=int(val.sum()), f1=None, auc=None)
            continue
        idv = np.where(val)[0]
        cols = CRITERION_FEATURES.get(name, QUALITY_FEATURES)
        src = source_of(name)
        base, kind = parse_source(src)

        if base == "cnn" or not cols:
            stack_v[idv, j] = oof_v[idv, j]
            results["violations"][name] = _score(yv[idv], oof_v[idv, j])
            results["comparison"][name + "_cnn_only"] = results["violations"][name]
            results["comparison"][name + "_fused"] = results["violations"][name]
            continue

        Xv = data[cols].fillna(0).values if cols else np.zeros((n, 0), dtype=float)
        cnn_v = _cv_cnn_only(oof_v[idv, j], yv[idv].astype(int), groups[idv])
        if base == "geo":
            model_v = _cv_geo_only(Xv[idv], yv[idv].astype(int), groups[idv],
                                   kind=kind)
        else:
            model_v = _cv_fuse(Xv[idv], oof_v[idv, j], yv[idv].astype(int),
                               groups[idv], kind=kind)
        stack_v[idv, j] = model_v
        results["violations"][name] = _score(yv[idv], model_v)
        results["comparison"][name + "_cnn_only"] = _score(yv[idv], cnn_v)
        results["comparison"][name + "_fused"] = results["violations"][name]
    return stack_q, stack_v, results


def fit_final_stackers(data: pd.DataFrame, oof_v: np.ndarray) -> dict:
    """Обучить финальные стекеры на всех данных и сохранить их (для инференса)."""
    real = real_mask(data)
    n = len(data)
    stackers = {}
    for j, name in enumerate(C.VIOLATIONS):
        yv = data["viol_" + name].values.astype(float)
        val = ~np.isnan(yv) & ~np.isnan(oof_v[:, j]) & real
        cols = CRITERION_FEATURES.get(name, QUALITY_FEATURES)
        src = source_of(name)
        base, kind = parse_source(src)
        if base not in ("fused", "geo") or not cols:
            continue
        if val.sum() and len(np.unique(yv[val])) > 1:
            Xv = data[cols].fillna(0).values if cols else np.zeros((n, 0), dtype=float)
            use_cnn = (base == "fused")
            base_X = (np.column_stack([oof_v[val, j], Xv[val]]) if use_cnn
                      else Xv[val])
            clf = fit_clf(kind, base_X, yv[val].astype(int))
            stackers[name] = dict(clf=clf, features=list(cols),
                                  use_cnn=use_cnn, kind=kind)
    try:
        import joblib

        joblib.dump(stackers, STACKER_PATH)
        print("[stack] стекеры сохранены:", STACKER_PATH)
    except Exception as e:  # joblib есть в requirements, но не критично
        print("[stack] не удалось сохранить стекеры:", e)
    return stackers


def main():
    manifest = load_manifest(C.MANIFEST_CSV)
    oof_q = np.load(C.OOF_QUALITY_NPY)
    oof_v = np.load(C.OOF_VIOLATION_NPY)

    data = feat_lib.load_features(manifest)
    real = real_mask(data)
    if not real.all():
        print(f"[stack] синтетических строк исключено из оценки: {int((~real).sum())}")

    stack_q, stack_v, results = build_scores(data, oof_q, oof_v)
    np.save(C.STACK_OOF_VIOLATION_NPY, stack_v)

    fit_final_stackers(data, oof_v)

    with open(STACK_METRICS, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=float)

    print("\n=== Гибрид (CNN + геометрия) vs только CNN ===")
    c = results["comparison"]
    if c["quality_cnn_only"].get("f1") is not None:
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