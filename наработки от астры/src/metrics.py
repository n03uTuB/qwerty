# -*- coding: utf-8 -*-
"""Метрики, пороги и доверительные интервалы (без зависимости от torch).

Вынесены в отдельный модуль, чтобы :mod:`src.train`, :mod:`src.stack`,
:mod:`src.calibrate` и :mod:`src.evaluate` импортировали их без циклических
зависимостей и без обязательного наличия torch.

Метрика организатора — macro-F1 по 4 меткам (:data:`ORG_LABELS`), каждая метка
есть OR своих критериев. Порог выбирается одним из режимов config.THRESHOLD_MODE:

  * ``f1``    — порог, максимизирующий F1 на переданной части (дефолт, воспроизводит
                подтверждённый v3 macro-F1 0.563);
  * ``blend`` — 0.5*(prior + f1): устойчив к калибровке (fable, +0.017..+0.021);
  * ``prior`` — столько снимков, сколько нарушений ожидается по доле;
  * ``nested``— консервативно: при отсутствии групп откатывается к ``blend``.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from . import config as C

ORG_LABELS: Dict[str, Sequence[str]] = C.ORG_LABELS
ORG_LABEL_NAMES = list(ORG_LABELS.keys())


# --------------------------------------------------------------------------- #
# Пороги
# --------------------------------------------------------------------------- #
def best_threshold(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    """Порог, максимизирующий F1 (сетка 0.05..0.95). -> (порог, F1)."""
    from sklearn.metrics import f1_score

    ths = np.linspace(0.05, 0.95, 91)
    best, bt = -1.0, 0.5
    for t in ths:
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best:
            best, bt = f, float(t)
    return bt, float(best)


def prior_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Порог по ожидаемой доле нарушений (самая устойчивая величина при 6-36 pos)."""
    prevalence = float(np.mean(y)) if len(y) else 0.0
    if prevalence <= 0 or prevalence >= 1 or len(p) == 0:
        return 0.5
    count = max(1, int(round(prevalence * len(p))))
    return float(np.sort(p)[::-1][min(count, len(p)) - 1])


def blend_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """0.5*(prior + f1) — устойчивый к калибровке дефолт (fable_solution)."""
    return 0.5 * (prior_threshold(y, p) + best_threshold(y, p)[0])


def pick_threshold(y: np.ndarray, p: np.ndarray,
                   mode: Optional[str] = None) -> Tuple[float, float]:
    """Выбрать порог по режиму и вернуть (порог, F1 на этой же части)."""
    from sklearn.metrics import f1_score

    mode = mode or getattr(C, "THRESHOLD_MODE", "f1")
    if mode == "prior":
        thr = prior_threshold(y, p)
    elif mode == "blend":
        thr = blend_threshold(y, p)
    elif mode == "nested":
        # Настоящая вложенная схема требует групп и реализована в calibrate.py;
        # здесь безопасный откат к blend.
        thr = blend_threshold(y, p)
    else:
        return best_threshold(y, p)
    return float(thr), float(f1_score(y, (p >= thr).astype(int), zero_division=0))


# --------------------------------------------------------------------------- #
# Метрики
# --------------------------------------------------------------------------- #
def _metric_value(y, p, metric: str, thr: float = 0.5):
    from sklearn.metrics import (average_precision_score, f1_score,
                                 roc_auc_score)

    try:
        if metric == "f1":
            return float(f1_score(y, (p >= thr).astype(int), zero_division=0))
        if metric == "auc":
            return float(roc_auc_score(y, p))
        if metric == "ap":
            return float(average_precision_score(y, p))
    except Exception:
        return None
    return None


def bootstrap_ci(y, p, metric: str = "auc", n: int = 1000, seed: int = 0,
                 thr: float = 0.5) -> Tuple[float, float]:
    """95% ДИ бутстрэпом по объектам (случайный выбор с возвращением)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    N = len(y)
    if N == 0:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2:
            continue
        v = _metric_value(yy, pp, metric, thr)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def bootstrap_ci_by_study(y, p, groups, metric: str = "auc", n: int = 1000,
                          seed: int = 0, thr: float = 0.5) -> Tuple[float, float]:
    """95% ДИ бутстрэпом по ИССЛЕДОВАНИЯМ (кластер-бутстрэп).

    Объектный бутстрэп занижает ДИ, т.к. снимки одного пациента коррелируют.
    Здесь resample идёт по study_uid, а внутри — берутся все его снимки.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    if len(uniq) == 0:
        return (float("nan"), float("nan"))
    by_group = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_group[g] for g in pick])
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2:
            continue
        v = _metric_value(yy, pp, metric, thr)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


# --------------------------------------------------------------------------- #
# Метка организатора
# --------------------------------------------------------------------------- #
def org_label_vector(viol: np.ndarray, region_idx: np.ndarray,
                     name: str) -> np.ndarray:
    """Собрать бинарный вектор метки организатора по мультилейбл-вероятностям.

    ``viol`` — (N, 5) вероятности критериев; метка = max по входящим критериям
    (только применимым к области снимка). Так «укладка» = OR укладки позвоночника
    и позиционирования бедра.
    """
    crit = ORG_LABELS[name]
    out = np.zeros(len(viol), dtype=np.float32)
    for i, c in enumerate(crit):
        j = C.VIOLATION_IDX[c]
        v = viol[:, j]
        out = v if i == 0 else np.maximum(out, v)
    return out


def org_label_truth(df, name: str) -> np.ndarray:
    """Истина метки организатора из колонок viol_* (NaN -> 0, как OR неприменимых)."""
    crit = ORG_LABELS[name]
    out = np.zeros(len(df), dtype=np.float32)
    for i, c in enumerate(crit):
        col = "viol_" + c
        if col not in df:
            continue
        v = np.nan_to_num(df[col].values.astype(float), nan=0.0)
        out = v if i == 0 else np.maximum(out, v)
    return out


def organizer_macro_f1(truths: Dict[str, np.ndarray],
                       preds: Dict[str, np.ndarray],
                       thresholds: Optional[Dict[str, float]] = None
                       ) -> Tuple[float, Dict[str, float]]:
    """macro-F1 по 4 меткам организатора -> (macro, per_label F1)."""
    from sklearn.metrics import f1_score

    thresholds = thresholds or {}
    per = {}
    for name in ORG_LABEL_NAMES:
        y = np.asarray(truths[name])
        p = np.asarray(preds[name])
        if len(y) == 0 or len(np.unique(y)) < 2:
            per[name] = float("nan")
            continue
        thr = thresholds.get(name, 0.5)
        per[name] = float(f1_score(y, (p >= thr).astype(int), zero_division=0))
    vals = [v for v in per.values() if np.isfinite(v)]
    return (float(np.mean(vals)) if vals else float("nan")), per