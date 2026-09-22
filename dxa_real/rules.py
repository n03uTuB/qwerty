"""Правила прямо из ТЗ: без обучения, без подбора порогов на данных.

ТЗ задаёт критерии численно, поэтому часть нарушений определяется измерением:
  * «правильно выровненная ось позвоночника (допустимый наклон до 5 градусов)»;
  * «по 3 см сверху и снизу от области интереса, 2 см от края правого и левого».
Такие правила ничего не подгоняют под выборку, их нельзя переобучить, и врачу
понятно, почему сработало.
"""

from __future__ import annotations

import numpy as np

from .data import LABEL_AXIS, LABEL_FOREIGN, LABEL_POSITIONING, LABEL_ROI

AXIS_LIMIT_DEG = 5.0       # ТЗ, п. 2.3
MARGIN_TOP_CM = 3.0        # ТЗ, п. 2.3
MARGIN_BOTTOM_CM = 3.0
MARGIN_SIDE_CM = 2.0


def spine_rules(features: dict) -> dict[str, tuple[int, str]]:
    result = {}
    angle = features.get("spine_midline_deg", 0.0)
    result[LABEL_AXIS] = (int(angle > AXIS_LIMIT_DEG),
                          f"наклон оси {angle:.1f}° при допуске {AXIS_LIMIT_DEG:g}°")
    foreign = features.get("foreign_count", 0.0)
    result[LABEL_FOREIGN] = (int(foreign >= 1),
                             f"объектов вне костных структур: {int(foreign)}")
    return result


def femur_rules(features: dict) -> dict[str, tuple[int, str]]:
    top = features.get("femur_margin_top_cm", 9.9)
    bottom = features.get("femur_margin_bottom_cm", 9.9)
    medial = features.get("femur_margin_medial_cm", 9.9)
    lateral = features.get("femur_margin_lateral_cm", 9.9)
    violated = (top < MARGIN_TOP_CM or bottom < MARGIN_BOTTOM_CM
                or medial < MARGIN_SIDE_CM or lateral < MARGIN_SIDE_CM)
    return {LABEL_ROI: (int(violated),
                        f"отступы: верх {top:.1f}, низ {bottom:.1f}, медиально {medial:.1f}, "
                        f"латерально {lateral:.1f} см при норме 3/3/2/2")}


def apply(features: dict, region: str) -> dict[str, tuple[int, str]]:
    return spine_rules(features) if region == "spine" else femur_rules(features)


def score_rule(y: np.ndarray, predicted: np.ndarray) -> dict:
    tp = int(((y == 1) & (predicted == 1)).sum())
    fn = int(((y == 1) & (predicted == 0)).sum())
    tn = int(((y == 0) & (predicted == 0)).sum())
    fp = int(((y == 0) & (predicted == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return dict(n=len(y), positives=int(y.sum()), tp=tp, fp=fp, fn=fn, tn=tn,
                sensitivity=recall, specificity=tn / max(tn + fp, 1),
                precision=precision,
                f1=2 * precision * recall / max(precision + recall, 1e-9))
