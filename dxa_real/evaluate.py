"""Честная оценка: повторная кросс-валидация с группировкой по пациенту.

Отличия от типичной ошибки «подобрали порог на тех же данных»:
  * порог выбирается во ВЛОЖЕННОЙ кросс-валидации внутри обучающей части;
  * разбиение — StratifiedGroupKFold по study_uid, повторяется с разными сидами;
  * доверительные интервалы — бутстрэп ПО ИССЛЕДОВАНИЯМ (снимки пациента зависимы).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import features as F
from .data import LABELS, Image

N_SPLITS = 5
SEEDS = (0, 1, 2, 3, 4)


def make_model():
    """Простая калиброванная модель: данных мало, сложная переобучится."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)),
    ])


def build_table(images: list[Image], feature_rows: list[dict] | None = None) -> pd.DataFrame:
    rows = feature_rows if feature_rows is not None else [F.compute(im) for im in images]
    table = pd.DataFrame(rows)
    table["study_uid"] = [im.study_uid for im in images]
    table["region"] = [im.region for im in images]
    table["quality"] = [im.quality for im in images]
    table["synthetic"] = [getattr(im, "synthetic", False) for im in images]
    for label in LABELS:
        table[label] = [im.labels.get(label, np.nan) for im in images]
    return table


def feature_columns(region: str, label: str | None = None, use_meta: bool = False) -> list[str]:
    """Под каждый критерий — свой небольшой набор признаков.

    При 6–36 положительных примерах модель на двух десятках признаков переобучается:
    одиночный угол оси давал AUC 0.87, а модель на всех признаках — 0.67.
    """
    if label is not None and (region, label) in F.CRITERION_FEATURES:
        columns = list(F.CRITERION_FEATURES[(region, label)])
    else:
        columns = list(F.SPINE_KEYS if region == "spine" else F.FEMUR_KEYS)
    if use_meta:  # найденный сигнал: число копий снимка в выгрузке
        columns += ["copies", "n_images"]
    return columns


def _best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def _prior_threshold(y_train: np.ndarray, p_valid: np.ndarray) -> float:
    """Порог по ожидаемой доле нарушений.

    При 6–10 положительных примерах порог, максимизирующий F1, скачет от фолда к фолду.
    Устойчивее назначить нарушением столько снимков, сколько их ожидается по обучающей
    выборке: доля нарушений там — самая надёжная величина, которая у нас есть.
    """
    prevalence = float(np.mean(y_train))
    if prevalence <= 0 or prevalence >= 1 or len(p_valid) == 0:
        return 0.5
    count = max(1, int(round(prevalence * len(p_valid))))
    return float(np.sort(p_valid)[::-1][min(count, len(p_valid)) - 1])


def cross_validate(X: np.ndarray, y: np.ndarray, groups: np.ndarray, *, train_mask: np.ndarray | None = None,
                   seeds=SEEDS, n_splits: int = N_SPLITS,
                   threshold_mode: str = "nested") -> tuple[np.ndarray, np.ndarray]:
    """OOF-вероятности и OOF-предсказания с порогом из вложенной CV.

    train_mask помечает объекты, которые можно использовать ТОЛЬКО для обучения
    (например, синтетические) — в оценку они не попадают.
    """
    n = len(y)
    probs = np.zeros(n)
    preds = np.zeros(n)
    counts = np.zeros(n)
    usable = np.ones(n, bool) if train_mask is None else train_mask

    for seed in seeds:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        eval_idx = np.where(usable)[0]
        for tr_rel, va_rel in splitter.split(X[eval_idx], y[eval_idx], groups=groups[eval_idx]):
            va = eval_idx[va_rel]
            va_groups = set(groups[va])
            # в обучение берём всё, кроме исследований валидации (включая синтетику)
            tr = np.array([i for i in range(n) if groups[i] not in va_groups])
            if len(np.unique(y[tr])) < 2:
                continue
            model = make_model().fit(X[tr], y[tr])
            fold_probs = model.predict_proba(X[va])[:, 1]
            probs[va] += fold_probs

            if threshold_mode == "prior":
                threshold = _prior_threshold(y[tr], fold_probs)
                preds[va] += (fold_probs >= threshold).astype(float)
                counts[va] += 1
                continue

            # порог — по внутренней CV на обучающей части, без подглядывания в валидацию
            inner_probs, inner_y = [], []
            inner = StratifiedGroupKFold(n_splits=min(4, len(np.unique(groups[tr]))), shuffle=True,
                                         random_state=seed)
            try:
                for itr_rel, iva_rel in inner.split(X[tr], y[tr], groups=groups[tr]):
                    itr, iva = tr[itr_rel], tr[iva_rel]
                    if len(np.unique(y[itr])) < 2:
                        continue
                    inner_model = make_model().fit(X[itr], y[itr])
                    inner_probs.append(inner_model.predict_proba(X[iva])[:, 1])
                    inner_y.append(y[iva])
            except ValueError:
                pass
            threshold = _best_threshold(np.concatenate(inner_y), np.concatenate(inner_probs)) \
                if inner_probs else 0.5
            preds[va] += (model.predict_proba(X[va])[:, 1] >= threshold).astype(float)
            counts[va] += 1

    valid = counts > 0
    probs[valid] /= counts[valid]
    preds[valid] = (preds[valid] / counts[valid] >= 0.5).astype(float)
    probs[~valid] = np.nan
    preds[~valid] = np.nan
    return probs, preds


def bootstrap_ci(y: np.ndarray, p: np.ndarray, groups: np.ndarray, metric: str = "auc",
                 n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Бутстрэп по исследованиям: снимки одного пациента ресэмплятся вместе."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    index = {g: np.where(groups == g)[0] for g in unique}
    values = []
    for _ in range(n):
        picked = np.concatenate([index[g] for g in rng.choice(unique, len(unique))])
        yy, pp = y[picked], p[picked]
        if len(np.unique(yy)) < 2:
            continue
        values.append(roc_auc_score(yy, pp) if metric == "auc" else average_precision_score(yy, pp))
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


@dataclass
class Score:
    name: str
    n: int
    positives: int
    auc: float
    auc_ci: tuple[float, float]
    ap: float
    f1: float
    sensitivity: float
    specificity: float
    extra: dict = field(default_factory=dict)

    def row(self) -> str:
        lo, hi = self.auc_ci
        return (f"{self.name:34} n={self.n:4d} поз={self.positives:3d}  "
                f"AUC={self.auc:.3f} [{lo:.2f};{hi:.2f}]  AP={self.ap:.3f}  "
                f"F1={self.f1:.3f}  чувст={self.sensitivity:.2f}  спец={self.specificity:.2f}")


def score(name: str, y: np.ndarray, probs: np.ndarray, preds: np.ndarray, groups: np.ndarray) -> Score:
    tp = int(((y == 1) & (preds == 1)).sum())
    fn = int(((y == 1) & (preds == 0)).sum())
    tn = int(((y == 0) & (preds == 0)).sum())
    fp = int(((y == 0) & (preds == 1)).sum())
    return Score(
        name=name, n=len(y), positives=int(y.sum()),
        auc=float(roc_auc_score(y, probs)), auc_ci=bootstrap_ci(y, probs, groups),
        ap=float(average_precision_score(y, probs)),
        f1=float(f1_score(y, preds, zero_division=0)),
        sensitivity=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1),
    )


def evaluate(images: list[Image], table: pd.DataFrame | None = None, verbose: bool = True,
             use_meta: bool = False, threshold_mode: str = "nested") -> dict:
    """Метрики по каждому критерию и итоговому качеству, отдельно по областям."""
    table = build_table(images) if table is None else table
    results: dict[str, Score] = {}
    label_predictions: dict[tuple[str, str], np.ndarray] = {}

    for region in ("spine", "femur"):
        region_mask = (table["region"] == region).values
        groups_all = table.loc[region_mask, "study_uid"].values
        synthetic = table.loc[region_mask, "synthetic"].values.astype(bool)

        for label in LABELS:
            X_all = table.loc[region_mask, feature_columns(region, label, use_meta)].fillna(0).values
            y_all = table.loc[region_mask, label].values.astype(float)
            known = ~np.isnan(y_all)
            if known.sum() == 0 or len(np.unique(y_all[known])) < 2:
                continue
            real = known & ~synthetic
            if real.sum() == 0 or len(np.unique(y_all[real])) < 2:
                continue
            X, y, groups = X_all[known], y_all[known].astype(int), groups_all[known]
            train_mask = (~synthetic[known])
            probs, preds = cross_validate(X, y, groups, train_mask=train_mask,
                                          threshold_mode=threshold_mode)
            evaluated = ~np.isnan(probs) & train_mask
            results[f"{region}: {label}"] = score(f"{region}: {label}", y[evaluated], probs[evaluated],
                                                  preds[evaluated], groups[evaluated])
            full = np.full(region_mask.sum(), np.nan)
            full[np.where(known)[0][evaluated]] = preds[evaluated]
            label_predictions[(region, label)] = full

        # итоговое качество: (а) прямая модель, (б) ИЛИ по критериям
        X_all = table.loc[region_mask, feature_columns(region, None, use_meta)].fillna(0).values
        y_all = table.loc[region_mask, "quality"].values.astype(float)
        known = ~np.isnan(y_all) & ~synthetic
        if known.sum() and len(np.unique(y_all[known])) > 1:
            X, y, groups = X_all[known], y_all[known].astype(int), groups_all[known]
            probs, preds = cross_validate(X, y, groups, threshold_mode=threshold_mode)
            results[f"{region}: ИТОГ качество (модель)"] = score(
                f"{region}: ИТОГ качество (модель)", y, probs, preds, groups)

            stacked = [label_predictions[(region, label)][known] for (r, label) in label_predictions if r == region]
            if stacked:
                union = np.nanmax(np.vstack(stacked), axis=0)
                union = np.nan_to_num(union, nan=0.0)
                results[f"{region}: ИТОГ качество (ИЛИ критериев)"] = score(
                    f"{region}: ИТОГ качество (ИЛИ критериев)", y, union, union, groups)

    macro = float(np.mean([s.f1 for name, s in results.items() if "ИТОГ" not in name]))
    if verbose:
        print(f"\n{'=' * 100}")
        for name in sorted(results):
            print(results[name].row())
        print(f"\nmacro-F1 по критериям: {macro:.3f}")
    return {"scores": results, "macro_f1": macro}
