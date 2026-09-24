# -*- coding: utf-8 -*-
"""Перебор геометрических признаков для критериев «предметы» и «укладка».

Запуск (когда появится датасет):

    python scripts/sweep_criteria.py --manifest artifacts/manifest.csv
    python scripts/sweep_criteria.py --target spine_artifacts --top 15

Что делает:
  1. грузит манифест, строит признаки (при необходимости) и атлас-шаблон;
  2. для критерия сравнивает базовый набор (из ``config.CRITERION_FEATURES_OVERRIDE``)
     с жадным прямым отбором признаков по GroupKFold (группы = ``study_uid``);
  3. печатает ROC-AUC / macro-F1 и лучший набор — его можно вписать в
     ``CRITERION_FEATURES_OVERRIDE``.

Метрика — честная: CV по группам-исследованиям, порог 0.5, без утечки. Редкие
классы (6..17 положительных) делают macro-F1 шумным, поэтому отбор ведётся по
AUC, а macro-F1 показывается справочно.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config as C  # noqa: E402
from src import features as F  # noqa: E402
from src.data import load_manifest  # noqa: E402

TARGET_REGION = {
    "spine_artifacts": C.REGION_SPINE,
    "spine_positioning": C.REGION_SPINE,
    "femur_positioning": None,
    "femur_roi": None,
}


def _clf():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)),
    ])


def cv_score(X, y, groups, n_splits=5):
    """Средние ROC-AUC и macro-F1 по GroupKFold (порог 0.5)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    oof = np.full(len(y), np.nan)
    for tr, va in GroupKFold(n_splits=n_splits).split(X, groups=groups):
        if len(np.unique(y[tr])) < 2:
            continue
        m = _clf().fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    ok = np.isfinite(oof)
    if len(np.unique(y[ok])) < 2 or ok.sum() < 5:
        return float("nan"), float("nan")
    auc = roc_auc_score(y[ok], oof[ok])
    f1 = f1_score(y[ok], (oof[ok] >= 0.5).astype(int), zero_division=0)
    return float(auc), float(f1)


def _candidates(df, base):
    names = [c for c in F.FEATURE_NAMES if c in df.columns]
    extra = [c for c in df.columns
             if c not in names and c.startswith(("art_", "atlas_", "align_"))]
    seen, out = set(), []
    for n in names + extra:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def greedy(X, y, groups, cands, max_k=6, eps=0.003, verbose=True):
    chosen, best = [], -1.0
    while len(chosen) < max_k:
        gain_name, gain_auc = None, best
        for c in cands:
            if c in chosen:
                continue
            auc, _ = cv_score(X[chosen + [c]], y, groups)
            if np.isfinite(auc) and auc > gain_auc + eps:
                gain_auc, gain_name = auc, c
        if gain_name is None:
            break
        chosen.append(gain_name)
        best = gain_auc
        if verbose:
            print("  + %-24s AUC=%.3f" % (gain_name, best))
    return chosen, best


def run_target(df, target, top=10, max_k=6):
    col = "viol_" + target
    if col not in df.columns:
        print("[%s] нет метки %s" % (target, col))
        return
    region = TARGET_REGION.get(target)
    sub = df.copy()
    if region is not None and "region" in sub.columns:
        sub = sub[sub["region"] == region]
    sub = sub[np.isfinite(pd.to_numeric(sub[col], errors="coerce"))]
    y = sub[col].astype(float).values
    if len(np.unique(y)) < 2 or y.sum() < 3:
        print("[%s] мало положительных (%d/%d) — перебор пропущен"
              % (target, int(y.sum()), len(y)))
        return
    groups = sub["study_uid"].values if "study_uid" in sub.columns else np.arange(len(sub))
    cands = _candidates(sub, None)
    X = sub[cands].astype(float).fillna(0.0)

    print("\n=== %s: положительных %d / %d ===" % (target, int(y.sum()), len(y)))
    base = C.CRITERION_FEATURES_OVERRIDE.get(target)
    if base:
        base = [c for c in base if c in cands]
        if base:
            auc, f1 = cv_score(X[base], y, groups)
            print("  базовый набор %s -> AUC=%.3f  macroF1=%.3f" % (base, auc, f1))

    # одиночные признаки
    singles = []
    for c in cands:
        auc, f1 = cv_score(X[[c]], y, groups)
        if np.isfinite(auc):
            singles.append((abs(auc - 0.5) + 0.5, auc, f1, c))
    singles.sort(reverse=True)
    print("  топ одиночных признаков (по |AUC-0.5|):")
    for _, auc, f1, c in singles[:top]:
        print("    %-24s AUC=%.3f  macroF1=%.3f" % (c, auc, f1))

    print("  жадный отбор:")
    chosen, best = greedy(X, y, groups, [s[3] for s in singles[:25]], max_k=max_k)
    if chosen:
        auc, f1 = cv_score(X[chosen], y, groups)
        print("  ЛУЧШИЙ набор: %s -> AUC=%.3f  macroF1=%.3f" % (chosen, auc, f1))
        print("  впишите в config.CRITERION_FEATURES_OVERRIDE['%s']" % target)


def main():
    ap = argparse.ArgumentParser(description="Перебор признаков критериев")
    ap.add_argument("--manifest", default=C.MANIFEST_CSV)
    ap.add_argument("--target", default="both",
                    choices=["spine_artifacts", "spine_positioning", "both"])
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--max-k", type=int, default=6)
    ap.add_argument("--no-atlas", action="store_true")
    ap.add_argument("--real-only", action="store_true",
                    help="считать метрики только по реальным снимкам")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    if args.real_only and "synthetic" in manifest.columns:
        manifest = manifest[~manifest["synthetic"].fillna(False).astype(bool)]

    if not args.no_atlas:
        from src.atlas import Atlas, build_atlas
        if Atlas.load() is None:
            print("[sweep] атлас не найден — строю...")
            build_atlas(manifest)

    df = F.load_features(manifest)

    targets = (["spine_artifacts", "spine_positioning"] if args.target == "both"
               else [args.target])
    for t in targets:
        run_target(df, t, top=args.top, max_k=args.max_k)


if __name__ == "__main__":
    main()