# -*- coding: utf-8 -*-
"""Быстрый поиск признаков для самого слабого критерия — femur_positioning.

Работает по уже посчитанным artifacts/features.csv (без перечитывания DICOM).
Печатает:
  1) AUC каждого одиночного признака бедра против femur_positioning;
  2) AUC групповой CV-логистической регрессии для наборов-кандидатов.

    cd dxa_qc && python ../scripts/diag_femur_positioning.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dxa_qc"))

from src import config as C      # noqa: E402
from src import dataset as ds    # noqa: E402

CANDIDATES = {
    "A troch+area+neck (текущий)": ["femur_trochanter_bulge", "femur_troch_area_mm2",
                                     "femur_neck_width_mm"],
    "B troch+neck": ["femur_trochanter_bulge", "femur_neck_width_mm"],
    "E margins top+bottom": ["femur_margin_top_cm", "femur_margin_bottom_cm"],
    "E1 margins min+top+bottom": ["femur_margin_min_cm", "femur_margin_top_cm",
                                  "femur_margin_bottom_cm"],
    "E2 margin_min": ["femur_margin_min_cm"],
    "F troch+necktohead+neck": ["femur_trochanter_bulge", "femur_neck_to_head",
                                "femur_neck_width_mm"],
    "L margins+bulge": ["femur_margin_min_cm", "femur_margin_top_cm",
                        "femur_trochanter_bulge"],
    "M margins+shaftdeg": ["femur_margin_min_cm", "femur_margin_top_cm", "femur_shaft_deg"],
    "N shaftdeg+bulge": ["femur_shaft_deg", "femur_trochanter_bulge"],
    "O margins+width": ["femur_margin_min_cm", "femur_width_cm"],
    "P margin_min+top+bottom+shaftdeg": ["femur_margin_min_cm", "femur_margin_top_cm",
                                         "femur_margin_bottom_cm", "femur_shaft_deg"],
    "Q margins+axis": ["femur_margin_min_cm", "femur_margin_top_cm", "femur_axis_angle"],
    "K все признаки бедра": None,
}


def cv_auc(X, y, groups, seeds=(0, 1, 2, 3, 4)):
    probs = np.zeros(len(y))
    cnt = np.zeros(len(y))
    for seed in seeds:
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for trn, va in skf.split(X, y, groups):
            m = Pipeline([("s", StandardScaler()),
                          ("c", LogisticRegression(max_iter=3000, class_weight="balanced",
                                                   C=0.5))])
            m.fit(X[trn], y[trn])
            probs[va] += m.predict_proba(X[va])[:, 1]
            cnt[va] += 1
    ok = cnt > 0
    return roc_auc_score(y[ok], probs[ok] / cnt[ok]), ok


def main() -> None:
    manifest = ds.load_manifest(C.MANIFEST_CSV)
    feat = pd.read_csv(os.path.join(C.ARTIFACTS_DIR, "features.csv"))
    keep = ["image_uid"] + [c for c in feat.columns
                            if c not in ("study_uid", "region", "image_uid")]
    df = manifest.merge(feat[keep], on="image_uid", how="left")
    # только реальные снимки: синтетика тривиально отделима и завысила бы AUC
    df = df[df.region.str.contains("femur") & ~df.synthetic].copy()

    y = df["viol_femur_positioning"].values.astype(float)
    known = ~np.isnan(y)
    df, y = df[known], y[known].astype(int)
    groups = df["study_uid"].values
    print(f"бедро: n={len(df)}  нарушений={int(y.sum())}  "
          f"снимков/исследование~{len(df)/df.study_uid.nunique():.1f}\n")

    num = [c for c in df.columns if c.startswith("femur_")]
    print("=== одиночные признаки (AUC, сортировка по |AUC-0.5|) ===")
    singles = []
    for c in num:
        x = pd.to_numeric(df[c], errors="coerce").values
        ok = ~np.isnan(x)
        if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
            continue
        a = roc_auc_score(y[ok], x[ok])
        singles.append((max(a, 1 - a), a, c))
    for _, a, c in sorted(singles, reverse=True):
        print(f"    {c:32} AUC={a:.3f}")

    print("\n=== наборы-кандидаты (групповая CV-логистическая регрессия) ===")
    results = []
    for name, cols in CANDIDATES.items():
        cols = num if cols is None else [c for c in cols if c in df.columns]
        if not cols:
            print(f"    {name:34} пропущен (нет признаков)")
            continue
        X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
        a, _ = cv_auc(X, y, groups)
        results.append((a, name, len(cols)))
        print(f"    {name:34} AUC={a:.3f}  ({len(cols)} признаков)")
    print("\nлучший:", max(results)[1], f"AUC={max(results)[0]:.3f}")


if __name__ == "__main__":
    main()
