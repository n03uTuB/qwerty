# -*- coding: utf-8 -*-
"""Сквозная проверка сервиса dxa_qc на реальных данных.

1) Совпадают ли признаки из кэша сервиса (display LUT) с эталонным dxa_real.
2) Даёт ли геометрический стекер dxa_qc тот же прирост, что измерен в dxa_real.

    cd dxa_qc && python ../scripts/verify_dxa_qc_metrics.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
# корень набора данных: переменная окружения DXA_DATASET, иначе ../_dsroot
DATASET = os.environ.get("DXA_DATASET", os.path.join(HERE, "..", "_dsroot"))
sys.path.insert(0, os.path.join(HERE, "..", "dxa_qc"))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))  # корень репозитория (dxa_real/dxa_qc)

from src import config as C                       # noqa: E402
from src import dataset as ds                     # noqa: E402
from src import stack as st                       # noqa: E402
from dxa_real import features as RF               # noqa: E402
from dxa_real.data import load_dataset            # noqa: E402

# dxa_qc -> dxa_real (имена различаются только для средней линии)
RENAME = {"spine_midline_angle": "spine_midline_deg"}

CRIT_LABEL = {
    "spine_positioning": "Некорректная укладка",
    "spine_axis": "Не выравнена ось позвоночника",
    "spine_artifacts": "Присутствуют посторонние предметы",
    "femur_positioning": "Некорректная укладка",
    "femur_roi": "Некорректная область интереса",
}


def compare_features(manifest: pd.DataFrame) -> None:
    """Сверить признаки сервиса с эталоном dxa_real.

    Сопоставляем по ХЭШУ ПИКСЕЛЕЙ, а не по image_uid: при дедупликации сервис и
    эталон могут выбрать разные файлы одной и той же картинки (разные SOPInstanceUID).
    """
    import hashlib

    from src import dicom_io

    feat = pd.read_csv(os.path.join(C.ARTIFACTS_DIR, "features.csv"))
    keep = ["image_uid"] + [c for c in feat.columns
                            if c not in ("study_uid", "region", "image_uid")]
    merged = manifest.merge(feat[keep], on="image_uid", how="inner")

    by_hash = {}
    for _, r in merged.iterrows():
        try:
            _ds, arr = dicom_io.read_dicom(r["source_path"])
        except Exception:
            continue
        if arr is None:
            continue
        h = hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()
        by_hash[h] = r

    keys = ["spine_midline_angle", "spine_iliac_signal", "spine_ribs_signal",
            "spine_vertebra_peaks", "femur_trochanter_bulge", "femur_troch_area_mm2",
            "femur_neck_width_mm", "femur_height_cm", "femur_bone_ratio",
            "femur_margin_min_cm"]
    worst, checked, worst_key = 0.0, 0, ""
    for im in load_dataset(DATASET):
        h = hashlib.md5(np.ascontiguousarray(im.array).tobytes()).hexdigest()
        r = by_hash.get(h)
        if r is None:
            continue
        ref = RF.compute(im)
        for k in keys:
            rk = RENAME.get(k, k)
            if k not in r or rk not in ref:
                continue
            a, b = float(r[k]), float(ref[rk])
            # пол в знаменателе: у эталона бывают точные нули (например, площадь
            # выступа 0.0), иначе относительная ошибка взрывается на пустом месте
            d = abs(a - b) / max(abs(b), 1.0)
            if d > worst:
                worst, worst_key = d, f"{k} qc={a:.4g} real={b:.4g}"
        checked += 1
    print(f"[features] сверено изображений: {checked}  "
          f"макс. относительное расхождение: {worst:.2e}  ({worst_key})")
    print("           ->", "СОВПАДАЮТ" if worst < 1e-3 else "РАСХОДЯТСЯ")


def geo_metrics(manifest: pd.DataFrame, feat: pd.DataFrame) -> None:
    # в features.csv тоже есть study_uid/region -> берём только признаки
    keep = ["image_uid"] + [c for c in feat.columns
                            if c not in ("study_uid", "region", "image_uid")]
    data = manifest.merge(feat[keep], on="image_uid", how="left")
    data = data[~data["synthetic"].astype(bool)]
    groups = data["study_uid"].values
    seeds = (0, 1, 2, 3, 4)

    def best_thr(y, p):
        grid = np.linspace(0.01, 0.99, 197)
        sc = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
        return float(grid[int(np.argmax(sc))])

    def prior_thr(ytr, pva):
        prev = float(np.mean(ytr))
        if not 0 < prev < 1 or len(pva) == 0:
            return 0.5
        k = max(1, int(round(prev * len(pva))))
        return float(np.sort(pva)[::-1][min(k, len(pva)) - 1])

    def cv(X, y, mode, groups):
        probs = np.zeros(len(y))
        preds = np.zeros(len(y))
        cnt = np.zeros(len(y))
        for seed in seeds:
            skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            for tr, va in skf.split(X, y, groups):
                m = Pipeline([("s", StandardScaler()),
                              ("c", LogisticRegression(max_iter=2000,
                                                       class_weight="balanced", C=0.5))])
                m.fit(X[tr], y[tr])
                p = m.predict_proba(X[va])[:, 1]
                probs[va] += p
                if mode == "prior":
                    preds[va] += (p >= prior_thr(y[tr], p)).astype(float)
                else:
                    preds[va] += (p >= 0.5).astype(float)
                cnt[va] += 1
        ok = cnt > 0
        probs[ok] /= cnt[ok]
        return probs, ok

    print("\n[dxa_qc] геометрический стекер на признаках сервиса (5x5 CV)")
    f1s = []
    for name, cols in st.CRITERION_FEATURES.items():
        region = "lumbar_spine" if name.startswith("spine") else None
        if name.startswith("spine"):
            sub = data[data.region == "lumbar_spine"]
        else:
            sub = data[data.region.isin(["proximal_femur_left", "proximal_femur_right"])]
        y = sub["viol_" + name].values.astype(float)
        known = ~np.isnan(y)
        X = sub.loc[known, cols].fillna(0).values
        yy = y[known].astype(int)
        g = sub.loc[known, "study_uid"].values
        probs, ok = cv(X, yy, "nested", g)
        # порог по F1 (вложенность проверена в dxa_real; здесь — сверка значений)
        thr = best_thr(yy[ok], probs[ok])
        pred = (probs[ok] >= thr).astype(int)
        auc = roc_auc_score(yy[ok], probs[ok])
        f1 = f1_score(yy[ok], pred, zero_division=0)
        f1s.append(f1)
        print(f"    {name:20} n={len(yy):3d} pos={int(yy.sum()):2d}  AUC={auc:.3f}  F1={f1:.3f}")
    print(f"    {'macro-F1':20} {np.mean(f1s):.3f}")


def main() -> None:
    manifest = ds.load_manifest(C.MANIFEST_CSV)
    feat = pd.read_csv(os.path.join(C.ARTIFACTS_DIR, "features.csv"))
    compare_features(manifest)
    geo_metrics(manifest, feat)


if __name__ == "__main__":
    main()
