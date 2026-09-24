# -*- coding: utf-8 -*-
"""Быстрая проверка: помогают ли синтетические артефакты v2 геометрическому
ранжированию критерия «предметы» (spine_artifacts).

Считаем признаки для синтетических строк (их нет в features.csv), затем честно
сравниваем геомодель с синтетикой и без (порог по train-части, метрики по реальным).

Запуск:
    python scripts/exp_synth_art.py --per-type 60
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dxa_qc"))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import mlab as M  # noqa: E402
from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402

GEO = ["spine_ribs_signal", "spine_vertebra_peaks", "spine_midline_residual"]


def lr_factory(C_=0.5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return lambda: Pipeline([("s", StandardScaler()),
                             ("c", LogisticRegression(max_iter=2000,
                                                      class_weight="balanced", C=C_))])


def feats_for(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        arr = fl._load_for_features(r)
        spacing = (float(r["spacing_x"]), float(r["spacing_y"])) \
            if pd.notna(r.get("spacing_x")) else C.PIXEL_SPACING_MM
        f = fl.compute_features(arr, spacing, region=r.get("region"),
                                copies=float(r.get("copies", 1.0)))
        f["image_uid"] = r["image_uid"]
        f["study_uid"] = r["study_uid"]
        f["region"] = r.get("region")
        rows.append(f)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="out/manifest_art2.csv")
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    base_feat = pd.read_csv(os.path.join(C.ARTIFACTS_DIR, "features.csv"))
    have = set(base_feat["image_uid"])
    missing = man[~man["image_uid"].isin(have)].copy()
    print(f"строк манифеста: {len(man)}, без признаков: {len(missing)}")
    if len(missing):
        newf = feats_for(missing)
        base_feat = pd.concat([base_feat, newf], ignore_index=True)
    data = man.merge(base_feat, on="image_uid", how="left", suffixes=("", "_f"))
    if "study_uid_f" in data.columns:
        data["study_uid"] = data["study_uid"].fillna(data["study_uid_f"])

    real = np.asarray(ds.real_mask(data))
    synth_new = man["synthetic"].fillna(False).values & \
        data["viol_spine_artifacts"].fillna(0).values.astype(bool)
    print(f"реальных: {real.sum()}, синтетических артефактов: {synth_new.sum()}")

    y_raw = data["viol_spine_artifacts"].values.astype(float)
    spine = np.array([("spine" in str(r)) for r in data["region"].values])
    X = data[GEO].fillna(0).values
    groups = data["study_uid"].values

    # реальные строки как база; синтетика артефактов — только в train (по желанию)
    real_rows = real & spine & ~np.isnan(y_raw)
    y = np.nan_to_num(y_raw, nan=0.0).astype(int)
    for use_synth in (False, True):
        train_mask = real_rows.copy()
        if use_synth:
            train_mask = train_mask | synth_new
        probs, preds, counts = M.cv_oof(
            X, y, groups, lr_factory(0.5),
            seeds=range(5), train_mask=train_mask)
        yy = y_raw[real_rows]
        pp = probs[real_rows]
        auc = roc_auc_score(yy.astype(int), pp)
        thr = M.prior_threshold(yy.astype(int), pp)
        pr = (pp >= thr).astype(int)
        from sklearn.metrics import f1_score
        f1 = f1_score(yy.astype(int), pr, zero_division=0)
        print(f"  synth={'ДА ' if use_synth else 'НЕТ'}  AUC={auc:.3f}  F1(prior)={f1:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
