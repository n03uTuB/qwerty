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
  * ``nested``— консервативно: при отсутствии групп откатывается к ``blend``;
  * ``label`` — пороги критериев подбираются так, чтобы максимизировать F1
                ИМЕННО меток организатора (OR-комбинаций), а не критериев
                поодиночке (:func:`fit_org_label_thresholds`). Так как метрика
                организатора считается по OR-меткам, это прямая оптимизация
                целевой метрики.
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
        # "f1" и "mapped". Режим "mapped" — ПО КРИТЕРИЮ, реализован в
        # pick_criterion_thresholds (здесь нет имени критерия), поэтому для
        # одиночной пары (y, p) он эквивалентен "f1".
        return best_threshold(y, p)
    return float(thr), float(f1_score(y, (p >= thr).astype(int), zero_division=0))


def pick_criterion_thresholds(df, probs: np.ndarray, mask: np.ndarray,
                              mode: Optional[str] = None,
                              by_criterion: Optional[Dict[str, str]] = None
                              ) -> Dict[str, Dict]:
    """Пороги ВСЕХ критериев ТЗ, подобранные на части ``mask``.

    ``mode="mapped"`` (дефолт :data:`config.THRESHOLD_MODE`) — режим берётся для
    каждого критерия из :data:`config.THRESHOLD_MODE_BY_CRITERION` (например,
    ``prior`` для редких критериев и ``f1`` там, где ``prior`` вырождается).
    Иначе — единый режим ``mode``.

    Возвращает ``{criterion: dict(threshold, f1, mode, n, pos)}``.
    """
    mode = mode or getattr(C, "THRESHOLD_MODE", "f1")
    if by_criterion is None:
        by_criterion = getattr(C, "THRESHOLD_MODE_BY_CRITERION", {})
    mask = np.asarray(mask, dtype=bool)
    probs = np.asarray(probs)
    out: Dict[str, Dict] = {}
    for i, name in enumerate(C.VIOLATIONS):
        y = df["viol_" + name].values.astype(float)[mask]
        p = probs[mask, i].astype(float)
        ok = ~np.isnan(y) & np.isfinite(p)
        mo = by_criterion.get(name, "prior") if mode == "mapped" else mode
        if ok.sum() and len(np.unique(y[ok])) > 1:
            t, f = pick_threshold(y[ok], p[ok], mode=mo)
        else:
            t, f = 0.5, None
        out[name] = dict(threshold=float(t),
                         f1=(float(f) if f is not None else None),
                         mode=mo, n=int(ok.sum()),
                         pos=int((y[ok] > 0.5).sum()) if ok.sum() else 0)
    return out


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


def _best_f1_threshold(y: np.ndarray, p: np.ndarray,
                       grid: np.ndarray) -> float:
    """Порог из сетки, максимизирующий F1 (внутренний помощник)."""
    from sklearn.metrics import f1_score

    best, bt = -1.0, float(grid[len(grid) // 2])
    for t in grid:
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best:
            best, bt = f, float(t)
    return bt


def org_label_score(df, probs: np.ndarray, name: str):
    """Скор метки организатора по вероятностям критериев.

    Для каждой метки берётся максимум вероятностей ЕЁ критериев, но только там,
    где критерий применим (истина критерия определена, т.е. не NaN). Возвращает
    (score, applicable): ``applicable`` — маска снимков, где метка измерима.

    Пример: «укладка» = max(spine_positioning, femur_positioning), причём на
    снимках позвоночника применим только первый, на снимках бедра — только второй.
    """
    crits = ORG_LABELS[name]
    n = len(df)
    score = np.full(n, -np.inf, dtype=float)
    applicable = np.zeros(n, dtype=bool)
    for c in crits:
        j = C.VIOLATION_IDX[c]
        p = np.asarray(probs)[:, j].astype(float)
        col = ("viol_" + c)
        ok = np.isfinite(p)
        if col in df:
            ok = ok & ~np.isnan(df[col].values.astype(float))
        score = np.where(ok, np.maximum(score, p), score)
        applicable |= ok
    return score, applicable


def fit_org_label_thresholds(df, probs: np.ndarray, mask: np.ndarray,
                             grid: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Пороги критериев, максимизирующие F1 меток организатора на части ``mask``.

    В отличие от подбора порога под каждый критерий поодиночке, здесь целевая
    функция — F1 именно OR-метки организатора (укладка/ось/предметы/ROI).
    Для однометочных критериев (ось, предметы, ROI) порог = оптимальный порог
    метки; для «укладки» порог общий для двух критериев и подобран по объединению
    (критерии применимы на непересекающихся областях).

    Возвращает ``{criterion: threshold}`` для всех критериев ТЗ.
    """
    if grid is None:
        grid = np.linspace(0.05, 0.95, 91)
    thr = {c: 0.5 for c in C.VIOLATIONS}
    for name in ORG_LABEL_NAMES:
        score, applicable = org_label_score(df, probs, name)
        y = org_label_truth(df, name)
        m = np.asarray(mask, dtype=bool) & applicable & np.isfinite(score)
        if m.sum() < 2 or len(np.unique(y[m])) < 2:
            continue
        t = _best_f1_threshold(y[m], score[m], grid)
        for c in ORG_LABELS[name]:
            thr[c] = float(t)
    return thr


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