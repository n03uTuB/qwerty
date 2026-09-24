# -*- coding: utf-8 -*-
"""Общий честный CV-фреймворк для отбора моделей (геометрия и CNN).

Принципы (совпадают с dxa_real/evaluate.py и dxa_qc-стендами):
  * разбиение StratifiedGroupKFold по study_uid — пациент не пересекает фолды;
  * порог классификации выбирается ВНУТРИ обучающей части (nested) либо по
    ожидаемой доле нарушений (prior) — без подглядывания в валидацию;
  * метрики — по реальным снимкам; синтетика (если есть) идёт только в обучение;
  * повтор с несколькими сидами и усреднение OOF-вероятностей.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

SEEDS = (0, 1, 2, 3, 4)
N_SPLITS = 5


# --------------------------------------------------------------------------- #
# Пороги
# --------------------------------------------------------------------------- #
def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def prior_threshold(y: np.ndarray, p: np.ndarray) -> float:
    prevalence = float(np.mean(y)) if len(y) else 0.0
    if prevalence <= 0 or prevalence >= 1 or len(p) == 0:
        return 0.5
    count = max(1, int(round(prevalence * len(p))))
    return float(np.sort(p)[::-1][min(count, len(p)) - 1])


def pick_threshold(y: np.ndarray, p: np.ndarray, mode: str) -> float:
    if mode == "prior":
        return prior_threshold(y, p)
    return best_threshold(y, p)


# --------------------------------------------------------------------------- #
# OOF-предсказания
# --------------------------------------------------------------------------- #
def cv_oof(X, y, groups, factory, *, seeds=SEEDS, n_splits=N_SPLITS,
           threshold_mode="nested", train_mask=None):
    """OOF-вероятности и OOF-решения.

    train_mask: bool-массив, True там, где объект можно использовать ТОЛЬКО для
    обучения (синтетика). Метрика считается по объектам с train_mask=False.
    Возвращает (probs, preds, counts): probs/preds усреднены по сидам/фолдам.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    n = len(y)
    probs = np.zeros(n)
    preds = np.zeros(n)
    counts = np.zeros(n)
    usable = np.ones(n, bool) if train_mask is None else np.asarray(train_mask, bool)

    for seed in seeds:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        eval_idx = np.where(usable)[0]
        for tr_rel, va_rel in splitter.split(X[eval_idx], y[eval_idx], groups=groups[eval_idx]):
            va = eval_idx[va_rel]
            va_groups = set(groups[va])
            tr = np.array([i for i in range(n) if groups[i] not in va_groups])
            if len(np.unique(y[tr])) < 2:
                continue
            model = factory()
            model.fit(X[tr], y[tr])
            fold_probs = model.predict_proba(X[va])[:, 1]
            probs[va] += fold_probs

            if threshold_mode == "prior":
                thr = prior_threshold(y[tr], fold_probs)
                preds[va] += (fold_probs >= thr).astype(float)
                counts[va] += 1
                continue

            inner_probs, inner_y = [], []
            try:
                inner = StratifiedGroupKFold(n_splits=min(4, len(np.unique(groups[tr]))),
                                             shuffle=True, random_state=seed)
                for itr_rel, iva_rel in inner.split(X[tr], y[tr], groups=groups[tr]):
                    itr, iva = tr[itr_rel], tr[iva_rel]
                    if len(np.unique(y[itr])) < 2:
                        continue
                    im = factory()
                    im.fit(X[itr], y[itr])
                    inner_probs.append(im.predict_proba(X[iva])[:, 1])
                    inner_y.append(y[iva])
            except ValueError:
                pass
            thr = best_threshold(np.concatenate(inner_y), np.concatenate(inner_probs)) \
                if inner_probs else 0.5
            preds[va] += (fold_probs >= thr).astype(float)
            counts[va] += 1

    valid = counts > 0
    probs[valid] /= counts[valid]
    preds[valid] = (preds[valid] / counts[valid] >= 0.5).astype(float)
    probs[~valid] = np.nan
    preds[~valid] = np.nan
    return probs, preds, counts


# --------------------------------------------------------------------------- #
# Метрики
# --------------------------------------------------------------------------- #
def bootstrap_ci(y, p, groups, metric="auc", n=1000, seed=0):
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    index = {g: np.where(groups == g)[0] for g in unique}
    values = []
    for _ in range(n):
        picked = np.concatenate([index[g] for g in rng.choice(unique, len(unique))])
        yy, pp = y[picked], p[picked]
        if len(np.unique(yy)) < 2:
            continue
        values.append(roc_auc_score(yy, pp) if metric == "auc"
                      else average_precision_score(yy, pp))
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def metrics(y, probs, preds, groups=None, ci=False):
    y = np.asarray(y, dtype=int)
    probs = np.asarray(probs, dtype=float)
    preds = np.asarray(preds, dtype=float)
    ok = ~np.isnan(probs)
    y, probs, preds = y[ok], probs[ok], preds[ok]
    if groups is not None:
        groups = np.asarray(groups)[ok]
    out = dict(n=int(len(y)), pos=int(y.sum()))
    if len(y) == 0 or len(np.unique(y)) < 2:
        out.update(auc=float("nan"), ap=float("nan"), f1=float("nan"),
                   ba=float("nan"), sens=float("nan"), spec=float("nan"))
        return out
    out["auc"] = float(roc_auc_score(y, probs))
    out["ap"] = float(average_precision_score(y, probs))
    out["f1"] = float(f1_score(y, preds, zero_division=0))
    out["ba"] = float(balanced_accuracy_score(y, preds))
    tp = int(((y == 1) & (preds == 1)).sum())
    fn = int(((y == 1) & (preds == 0)).sum())
    tn = int(((y == 0) & (preds == 0)).sum())
    fp = int(((y == 0) & (preds == 1)).sum())
    out["sens"] = tp / max(tp + fn, 1)
    out["spec"] = tn / max(tn + fp, 1)
    if ci and groups is not None:
        out["auc_ci"] = bootstrap_ci(y, probs, groups, "auc")
    return out


def fmt(m, name="", width=34):
    lo, hi = m.get("auc_ci", (float("nan"), float("nan")))
    ci = f" [{lo:.2f};{hi:.2f}]" if not np.isnan(lo) else ""
    return (f"{name:{width}} AUC={m['auc']:.3f}{ci}  AP={m['ap']:.3f}  F1={m['f1']:.3f}  "
            f"BA={m['ba']:.3f}  ч={m['sens']:.2f} с={m['spec']:.2f}")
