# -*- coding: utf-8 -*-
"""Классификаторы гибридного стекера: LogReg / LDA-shrinkage / balanced bagging.

Вынесены в отдельный модуль, чтобы объекты корректно сериализовались в joblib.
``src.stack`` запускается как ``python -m src.stack`` (т.е. как ``__main__``), и
класс, определённый прямо в нём, записался бы как ``__main__.BalancedBag`` — а
при загрузке стекера из ``src.inference`` модуля ``__main__`` с этим классом уже
нет. Отдельный модуль делает путь класса импортируемым (``src.stack_clf``).

Источник скора по критерию (``config.CRITERION_SOURCES``) кодирует и модель, и
набор входов:

  * ``fused``/``geo``         — LogReg (вариант v3);
  * ``fused_lda``/``geo_lda`` — LDA со shrinkage (устойчив при 6-36 позитивах);
  * ``fused_bag``/``geo_bag`` — balanced bagging / EasyEnsemble (редкие классы).
"""
from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CLF_KINDS = ("logreg", "lda", "bag")
_SOURCE_SUFFIX = {"_lda": "lda", "_bag": "bag"}


def parse_source(src: str):
    """``'fused_bag' -> ('fused', 'bag')``; ``'geo' -> ('geo', 'logreg')``."""
    for suf, kind in _SOURCE_SUFFIX.items():
        if src.endswith(suf):
            return src[:-len(suf)], kind
    return src, "logreg"


def make_clf(kind: str = "logreg") -> Pipeline:
    if kind == "lda":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ])
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)),
    ])


class BalancedBag:
    """EasyEnsemble: N моделей на всех позитивах + случайной подвыборке негативов.

    Устойчив к сильному дисбалансу (например, ``femur_roi``: 7 позитивов из 150).
    """

    def __init__(self, n: int = 40, ratio: float = 3.0, seed: int = 0):
        self.n, self.ratio, self.seed = n, ratio, seed
        self.models_: list = []

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, int)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        if len(pos) == 0 or len(neg) == 0:
            self.models_ = [make_clf("logreg").fit(X, y)]
            return self
        rng = np.random.default_rng(self.seed)
        for _ in range(self.n):
            nn = min(len(neg), max(1, int(round(self.ratio * len(pos)))))
            idx = np.concatenate([pos, rng.choice(neg, nn, replace=False)])
            self.models_.append(make_clf("logreg").fit(X[idx], y[idx]))
        return self

    def predict_proba(self, X):
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.models_], axis=0)
        return np.column_stack([1 - p, p])


def fit_clf(kind: str, X, y):
    if kind == "bag":
        return BalancedBag().fit(X, y)
    return make_clf(kind).fit(X, y)
