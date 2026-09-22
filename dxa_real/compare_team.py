"""Пересчёт метрик решения команды по честной схеме.

Их OOF-предсказания (artifacts/oof_*.npy) сопоставляются с метками эксперта, после чего
считаются два варианта:

  «как в отчёте»  — порог подбирается максимизацией F1 на ТЕХ ЖЕ данных, по которым
                    потом считается F1. Это верхняя оценка, а не результат;
  «честно»        — порог подбирается на других исследованиях (по фолдам), к проверочным
                    применяется готовый. Доверительные интервалы — бутстрэп по пациентам.

Разница между ними и есть завышение от подбора порога.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from .data import LABEL_AXIS, LABEL_FOREIGN, LABEL_POSITIONING, LABEL_ROI, Image, load_dataset
from .evaluate import bootstrap_ci

# порядок колонок violation в их коде (config.VIOLATIONS)
TEAM_VIOLATIONS = ["spine_positioning", "spine_axis", "spine_artifacts",
                   "femur_positioning", "femur_roi"]
TEAM_TO_LABEL = {
    "spine_positioning": (LABEL_POSITIONING, "spine"),
    "spine_axis": (LABEL_AXIS, "spine"),
    "spine_artifacts": (LABEL_FOREIGN, "spine"),
    "femur_positioning": (LABEL_POSITIONING, "femur"),
    "femur_roi": (LABEL_ROI, "femur"),
}


def align(images: list[Image], features_csv: Path) -> list[Image | None]:
    """Их строка i -> наш снимок. Ключ — исследование и порядок съёмки внутри него.

    По SOPInstanceUID сопоставить нельзя: при дедупликации представителем одного и того же
    изображения может оказаться разный файл.
    """
    features = pd.read_csv(features_csv)
    by_study: dict[str, list[Image]] = {}
    for image in images:
        by_study.setdefault(image.study_uid, []).append(image)

    used: dict[str, int] = {}
    aligned: list[Image | None] = []
    for study_uid in features["study_uid"].astype(str):
        position = used.get(study_uid, 0)
        group = by_study.get(study_uid, [])
        aligned.append(group[position] if position < len(group) else None)
        used[study_uid] = position + 1
    return aligned


def _f1_at(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    return float(f1_score(y, (p >= threshold).astype(int), zero_division=0))


def _best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    grid = np.linspace(0.05, 0.95, 91)
    return float(grid[int(np.argmax([_f1_at(y, p, t) for t in grid]))])


def honest_f1(y: np.ndarray, p: np.ndarray, groups: np.ndarray, n_splits: int = 5) -> tuple[float, float, float]:
    """F1 с порогом, подобранным на других исследованиях. Возвращает (F1, чувств., спец.)."""
    predictions = np.zeros(len(y))
    splitter = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    for train, test in splitter.split(p.reshape(-1, 1), y, groups=groups):
        if len(np.unique(y[train])) < 2:
            continue
        threshold = _best_threshold(y[train], p[train])
        predictions[test] = (p[test] >= threshold).astype(float)
    tp = int(((y == 1) & (predictions == 1)).sum())
    fn = int(((y == 1) & (predictions == 0)).sum())
    tn = int(((y == 0) & (predictions == 0)).sum())
    fp = int(((y == 0) & (predictions == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return (2 * precision * recall / max(precision + recall, 1e-9), recall, tn / max(tn + fp, 1))


def evaluate_team(dataset_root: str | Path, artifacts: str | Path, verbose: bool = True) -> dict:
    artifacts = Path(artifacts)
    images = load_dataset(dataset_root)
    aligned = align(images, artifacts / "features.csv")

    oof_quality = np.load(artifacts / "oof_quality.npy")
    oof_violation = np.load(artifacts / "oof_violation.npy")
    stack_path = artifacts / "stack_oof_violation.npy"
    stacked = np.load(stack_path) if stack_path.is_file() else None

    rows = []
    for index, image in enumerate(aligned):
        if image is None or image.quality is None:
            continue
        rows.append(dict(index=index, image=image, study=image.study_uid, region=image.region,
                         quality=image.quality))
    if verbose:
        print(f"сопоставлено снимков: {len(rows)} из {len(aligned)}")

    results = {}

    def report(name, y, p, groups):
        if len(np.unique(y)) < 2:
            return
        reported_threshold = _best_threshold(y, p)
        reported = _f1_at(y, p, reported_threshold)
        fair, sensitivity, specificity = honest_f1(y, p, groups)
        auc = float(roc_auc_score(y, p))
        results[name] = dict(n=len(y), positives=int(y.sum()), auc=auc,
                             auc_ci=bootstrap_ci(y, p, groups), ap=float(average_precision_score(y, p)),
                             f1_reported=reported, f1_honest=fair,
                             sensitivity=sensitivity, specificity=specificity)

    # --- качество ---
    y = np.array([r["quality"] for r in rows], dtype=int)
    p = oof_quality[[r["index"] for r in rows]]
    groups = np.array([r["study"] for r in rows])
    report("качество (все области)", y, p, groups)
    for region in ("spine", "femur"):
        mask = np.array([r["region"] == region for r in rows])
        if mask.sum():
            report(f"качество ({region})", y[mask], p[mask], groups[mask])

    # --- типы нарушений ---
    for j, team_name in enumerate(TEAM_VIOLATIONS):
        label, region = TEAM_TO_LABEL[team_name]
        subset = [r for r in rows if r["region"] == region and label in r["image"].labels]
        if not subset:
            continue
        y = np.array([r["image"].labels[label] for r in subset], dtype=int)
        indices = [r["index"] for r in subset]
        # шкала инференса — гибридный стекер; где его значения нет, берём вероятность сети
        p = oof_violation[indices, j].astype(float)
        if stacked is not None:
            fused = stacked[indices, j]
            p = np.where(np.isnan(fused), p, fused)
        valid = ~np.isnan(p)
        report(f"нарушение: {team_name}", y[valid], p[valid],
               np.array([r["study"] for r in subset])[valid])

    violations = [v for k, v in results.items() if k.startswith("нарушение")]
    results["macro_f1_reported"] = float(np.mean([v["f1_reported"] for v in violations]))
    results["macro_f1_honest"] = float(np.mean([v["f1_honest"] for v in violations]))

    if verbose:
        print(f"\n{'показатель':34} {'n':>4} {'поз':>4} {'AUC':>7} {'95% ДИ':>16} "
              f"{'F1 как в отчёте':>16} {'F1 честно':>11}")
        for name, value in results.items():
            if not isinstance(value, dict):
                continue
            lo, hi = value["auc_ci"]
            print(f"{name:34} {value['n']:4d} {value['positives']:4d} {value['auc']:7.3f} "
                  f"{f'[{lo:.2f}; {hi:.2f}]':>16} {value['f1_reported']:16.3f} {value['f1_honest']:11.3f}")
        print(f"\nmacro-F1 по нарушениям: как в отчёте {results['macro_f1_reported']:.3f}, "
              f"честно {results['macro_f1_honest']:.3f}")
    return results
