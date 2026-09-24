# -*- coding: utf-8 -*-
"""Чистая (вложенная) проверка femur_positioning: cnn против геометрии.

Парный тест в exp_iter3.py показал +0.013 macro-F1 в пользу чистой геометрии,
но скоры геометрии там брались из общего 5-фолдового OOF (build_scores), то есть
для val-снимка внутренняя модель видела и другие val-снимки. Здесь это исключено:
для каждого внешнего фолда геометрия обучается ТОЛЬКО на train-части, порог
подбирается по внутренней CV внутри train, а на val переносится модель, обученная
на всём train. Источник CNN (база) остаётся прежним — общий OOF.

Метка «укладка» = OR(spine_positioning, femur_positioning). spine_positioning в
обоих вариантах одинаков, поэтому вся разница — в femur_positioning.

Запуск:
    cd dxa_qc && python ../scripts/check_femur_geo_clean.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dxa_qc"))

from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402
from src import stack as st        # noqa: E402
from src.train import best_threshold  # noqa: E402

import eval_honest_organizer as E  # noqa: E402

N_SEEDS = 20
CRIT = "femur_positioning"


def _clf():
    return Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(max_iter=2000,
                                              class_weight="balanced", C=1.0))])


def _inner_oof(X, y, groups, n_inner=5):
    oof = np.full(len(y), np.nan)
    for tr, va in GroupKFold(n_splits=n_inner).split(X, y, groups=groups):
        if len(np.unique(y[tr])) < 2:
            continue
        m = _clf()
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    return oof


def run(data, real, S, seed):
    idx = np.where(real)[0]
    g = data["study_uid"].values[idx]
    strat = data["region_idx"].values[idx] if "region_idx" in data.columns else None
    if strat is None or len(np.unique(strat)) < 2:
        splits = GroupKFold(n_splits=5).split(idx, groups=g)
    else:
        splits = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                      random_state=seed).split(idx, strat, groups=g)

    j_fp = C.VIOLATION_IDX[CRIT]
    cols = [c for c in st.CRITERION_FEATURES.get(CRIT, []) if c in data.columns]
    y_fp = data["viol_" + CRIT].values.astype(float)
    fem_mask = data["region"].str.contains("femur").values

    # срабатывания критериев по снимкам: [n, 2] — 0: spine_positioning, 1: femur_positioning
    fire_base = np.zeros((len(data), 2), dtype=int)   # femur -> cnn (общий OOF)
    fire_geo = np.zeros((len(data), 2), dtype=int)    # femur -> чистая геометрия
    for tr_i, va_i in splits:
        tr, va = idx[tr_i], idx[va_i]
        # spine_positioning: одинаково в обоих вариантах (CNN, общий OOF)
        j_sp = C.VIOLATION_IDX["spine_positioning"]
        rm_sp = data["region"].str.contains("spine").values
        y_sp = data["viol_spine_positioning"].values.astype(float)
        p_sp = S["cnn"][:, j_sp]
        trc = tr[rm_sp[tr] & ~np.isnan(y_sp[tr]) & ~np.isnan(p_sp[tr])]
        if len(trc) >= 2 and len(np.unique(y_sp[trc].astype(int))) >= 2:
            thr, _ = best_threshold(y_sp[trc].astype(int), p_sp[trc])
            ok = va[~np.isnan(p_sp[va])]
            fire_base[ok[p_sp[ok] >= thr], 0] = 1
            fire_geo[ok[p_sp[ok] >= thr], 0] = 1

        # femur_positioning: база — CNN общий OOF, кандидат — чистая вложенная геометрия
        p_cnn = S["cnn"][:, j_fp]
        trc = tr[fem_mask[tr] & ~np.isnan(y_fp[tr]) & ~np.isnan(p_cnn[tr])]
        if len(trc) >= 2 and len(np.unique(y_fp[trc].astype(int))) >= 2:
            thr, _ = best_threshold(y_fp[trc].astype(int), p_cnn[trc])
            ok = va[~np.isnan(p_cnn[va])]
            fire_base[ok[p_cnn[ok] >= thr], 1] = 1

        trg = tr[fem_mask[tr] & ~np.isnan(y_fp[tr])]
        vag = va[fem_mask[va] & ~np.isnan(y_fp[va])]
        if len(trg) >= 10 and len(vag) and len(np.unique(y_fp[trg].astype(int))) >= 2:
            Xtr = data.loc[trg, cols].fillna(0).values if cols else \
                np.zeros((len(trg), 1))
            Xva = data.loc[vag, cols].fillna(0).values if cols else \
                np.zeros((len(vag), 1))
            yt = y_fp[trg].astype(int)
            p_tr = _inner_oof(Xtr, yt, data["study_uid"].values[trg])
            ok = ~np.isnan(p_tr)
            if ok.sum() >= 5 and len(np.unique(yt[ok])) >= 2:
                thr, _ = best_threshold(yt[ok], p_tr[ok])
                m = _clf()
                m.fit(Xtr, yt)
                p_va = m.predict_proba(Xva)[:, 1]
                fire_geo[vag[p_va >= thr], 1] = 1

    y_sp = data["viol_spine_positioning"].values.astype(float)
    gt = (np.nan_to_num(y_sp, nan=0.0).astype(int) |
          np.nan_to_num(y_fp, nan=0.0).astype(int))
    out = {}
    for tag, fire in (("base", fire_base), ("geo", fire_geo)):
        pr = fire[:, 0] | fire[:, 1]
        out[tag] = dict(
            ukladka=f1_score(gt[idx], pr[idx], zero_division=0),
            femur=f1_score(np.nan_to_num(y_fp[idx], nan=0.0).astype(int),
                           fire[idx, 1], zero_division=0),
        )
    return out


def main() -> None:
    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    S = E.build_scores(data, real)
    print("Чистая вложенная проверка femur_positioning (источник геометрии —")
    print("только train-часть фолда). Метка «укладка» = OR(spine_pos, femur_pos).\n")
    res = [run(data, real, S, s) for s in range(N_SEEDS)]
    for tag in ("base", "geo"):
        uk = np.array([r[tag]["ukladka"] for r in res])
        fm = np.array([r[tag]["femur"] for r in res])
        print(f"  {tag:5} укладка {uk.mean():.3f} +- {uk.std():.3f}   "
              f"femur_positioning F1 {fm.mean():.3f} +- {fm.std():.3f}")
    d = np.array([r["geo"]["ukladka"] - r["base"]["ukladka"] for r in res])
    print(f"\n  разница «укладки» {d.mean():+.3f} +- {d.std():.3f}, "
          f"выигрыш в {int((d > 0).sum())}/{N_SEEDS} разбиений")
    print(f"  ожидаемая разница macro-F1: {d.mean() / 4:+.3f}")


if __name__ == "__main__":
    main()
