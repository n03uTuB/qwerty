# -*- coding: utf-8 -*-
"""Диагностика по каждому критерию: AUC vs AP vs F1@prior.

Почему это важно. В режиме порога ``prior`` порог ставится так, чтобы число
предсказанных нарушений равнялось ожидаемому (k = доля x n), то есть решение
целиком определяется тем, КОГО модель поставила в top-k. При этом

    F1 = 2*TP / (n_pos + k),

а поскольку n_pos и k фиксированы, F1@prior — монотонная функция числа истинных
срабатываний в top-k (precision@k). AUC же усредняет по всем порогам и к
верхушке списка слепа: у ``spine_axis`` AUC 0.889, а F1@prior всего 0.30.

Скрипт считает для каждого критерия и каждого источника (CNN / геометрия /
гибрид): AUC, Average Precision (AP), F1@prior, precision@k и 95% ДИ на
F1@prior бутстрэпом ПО ИССЛЕДОВАНИЯМ (снимки одного study зависимы).

Запуск (после train + stack):
    cd dxa_qc && python ../scripts/diag_criterion_metrics.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dxa_qc"))

from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402
from src import stack as st        # noqa: E402
from src.train import prior_threshold  # noqa: E402

N_BOOT = 2000
RNG = np.random.default_rng(0)


def _clf():
    return Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(max_iter=2000,
                                              class_weight="balanced", C=1.0))])


def _cv_score(X, y, groups, n_splits=5):
    """Честный OOF-скор логистической регрессии по геометрии."""
    oof = np.full(len(y), np.nan)
    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups=groups):
        if len(np.unique(y[tr])) < 2:
            continue
        m = _clf()
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    return oof


def _f1_at_prior(y, p):
    thr = prior_threshold(y, p)
    return float(f1_score(y, (p >= thr).astype(int), zero_division=0)), int((p >= thr).sum())


def _f1_ci_by_study(y, p, groups, n_boot=N_BOOT):
    """95% ДИ на F1@prior: ресэмплинг ИССЛЕДОВАНИЙ целиком (не снимков)."""
    studies = np.unique(groups)
    idx_by_study = {s: np.where(groups == s)[0] for s in studies}
    vals = []
    for _ in range(n_boot):
        pick = RNG.choice(studies, size=len(studies), replace=True)
        idx = np.concatenate([idx_by_study[s] for s in pick])
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2:
            continue
        vals.append(_f1_at_prior(yy, pp)[0])
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def _line(tag, y, p, groups):
    if p is None or np.all(np.isnan(p)):
        print(f"  {tag:10} нет данных")
        return None
    ok = ~np.isnan(p)
    yy, pp, gg = y[ok].astype(int), p[ok], groups[ok]
    if len(np.unique(yy)) < 2:
        print(f"  {tag:10} один класс")
        return None
    auc = roc_auc_score(yy, pp)
    ap = average_precision_score(yy, pp)
    f1, k = _f1_at_prior(yy, pp)
    lo, hi = _f1_ci_by_study(yy, pp, gg)
    thr = prior_threshold(yy, pp)
    tp = int(((yy == 1) & (pp >= thr)).sum())
    print(f"  {tag:10} AUC={auc:5.3f}  AP={ap:5.3f}  F1@prior={f1:5.3f} "
          f"[{lo:.3f},{hi:.3f}]  TP={tp:2d}/{k:2d}  pos={int(yy.sum()):2d}")
    return dict(auc=auc, ap=ap, f1=f1, k=k, tp=tp, lo=lo, hi=hi)


def main() -> None:
    manifest = ds.load_manifest(C.MANIFEST_CSV)
    data = fl.load_features(manifest)
    real = np.asarray(ds.real_mask(data))
    oof_cnn = np.load(os.path.join(C.ARTIFACTS_DIR, "oof_violation.npy"))
    fused = np.load(os.path.join(C.ARTIFACTS_DIR, "stack_oof_violation.npy"))
    if len(oof_cnn) != len(data):
        raise SystemExit(f"OOF ({len(oof_cnn)}) != манифест ({len(data)})")

    print("Порог prior = top-k, где k = доля x n. Цель — максимум precision@k.\n")
    summary = {}
    for j, name in enumerate(C.VIOLATIONS):
        region_key = "spine" if name.startswith("spine") else "femur"
        rmask = data["region"].str.contains(region_key).values & real
        y = data["viol_" + name].values.astype(float)
        known = ~np.isnan(y) & rmask
        if known.sum() == 0 or len(np.unique(y[known].astype(int))) < 2:
            continue
        yy = y[known].astype(int)
        g = data.loc[known, "study_uid"].values
        cols = [c for c in st.CRITERION_FEATURES.get(name, []) if c in data.columns]
        X = data.loc[known, cols].fillna(0).values if cols else np.zeros((len(yy), 1))
        cnn = oof_cnn[known, j]
        fus = fused[known, j]

        print(f"{name}  (n={len(yy)}, pos={int(yy.sum())}, признаки={cols})")
        res = {}
        res["cnn"] = _line("CNN", yy, cnn, g)
        res["geo"] = _line("геометрия", yy, _cv_score(X, yy, g), g)
        res["fused"] = _line("гибрид", yy, fus, g)
        if cols:
            # «сырое» правило: один семантически прямой признак без обучения
            best_raw = None
            for c in cols:
                v = data.loc[known, c].fillna(0).values
                if len(np.unique(v)) < 2:
                    continue
                for sign in (1.0, -1.0):
                    f1, k = _f1_at_prior(yy, sign * v)
                    if best_raw is None or f1 > best_raw[0]:
                        best_raw = (f1, k, c, sign)
            if best_raw:
                print(f"  {'сырой':10} F1@prior={best_raw[0]:5.3f}  "
                      f"TP={int(((yy==1)&((best_raw[3]*data.loc[known,best_raw[2]].fillna(0).values)>=prior_threshold(yy, best_raw[3]*data.loc[known,best_raw[2]].fillna(0).values))).sum()):2d}/{best_raw[1]:2d}"
                      f"  <- {best_raw[3]:+.0f}*{best_raw[2]}")
        summary[name] = res
        print()

    print("=" * 78)
    print("Сводка F1@prior (главный критерий отбора источника в режиме prior):")
    print(f"{'критерий':22} {'CNN':>7} {'геом':>7} {'гибрид':>7}   {'AP лучш.':>9} {'источник':>9}")
    for name, res in summary.items():
        f1s = {k: (v["f1"] if v else float("nan")) for k, v in res.items()}
        aps = {k: (v["ap"] if v else float("nan")) for k, v in res.items()}
        best_f1 = max(f1s, key=lambda k: f1s[k] if not np.isnan(f1s[k]) else -1)
        best_ap = max(aps, key=lambda k: aps[k] if not np.isnan(aps[k]) else -1)
        print(f"{name:22} {f1s['cnn']:7.3f} {f1s['geo']:7.3f} {f1s['fused']:7.3f}   "
              f"{aps[best_ap]:9.3f} {best_f1:>9}")


if __name__ == "__main__":
    main()
