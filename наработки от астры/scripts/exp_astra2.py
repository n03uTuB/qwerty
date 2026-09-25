# -*- coding: utf-8 -*-
"""Реализация проверяемых идей внешнего советника (Astra) для гибридного DXA-QC.

Идеи (нумерация Astra):
  #2 агрегация скоров по исследованию (метка бедра почти study-level);
  #3 регион-условный порог для метки «укладка»;
  #5 устойчивый классификатор для редких критериев (LDA-shrinkage);
  #6 прямая оптимизация macro-F1 организатора по порогам (coordinate ascent);
  #7 EasyEnsemble / balanced bagging для редких критериев.

Честная схема: StratifiedGroupKFold(5) по study_uid x N сидов; порог выбирается
ТОЛЬКО по train-части фолда; метрики только по реальным снимкам.

Два масштаба порога:
  * ``cnn``   — порог по train-пробам CNN (исторический harness решения);
  * ``clean`` — порог по OOF-скорам train-части (та же шкала, что и val) — честнее.

Печатает mean±std и пишет сводку в results/astra2_results.txt.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, f1_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from src import config as C          # noqa: E402
from src import stack as S           # noqa: E402
from src.data import load_manifest, real_mask   # noqa: E402
from src.features import load_features          # noqa: E402
from src.metrics import (ORG_LABEL_NAMES, blend_threshold,  # noqa: E402
                         pick_criterion_thresholds)

OUT_TXT = os.path.join(C.PROJECT_ROOT, "results", "astra2_results.txt")
_LINES: list = []


def log(msg: str) -> None:
    print(msg)
    _LINES.append(msg)


def flush() -> None:
    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(_LINES) + "\n")


# --------------------------------------------------------------------------- #
# Классификаторы и CV-скоры
# --------------------------------------------------------------------------- #
def make_clf(kind: str):
    if kind == "lda":
        return Pipeline([("s", StandardScaler()),
                         ("c", LinearDiscriminantAnalysis(solver="lsqr",
                                                          shrinkage="auto"))])
    return Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(max_iter=1000,
                                              class_weight="balanced"))])


class BalancedBag:
    """EasyEnsemble: N моделей на всех позитивах + случайных негативах."""

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


def cv_score(X, y, groups, kind: str, n_splits: int = 5) -> np.ndarray:
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    oof = np.full(len(y), np.nan)
    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups=groups):
        if len(np.unique(y[tr])) < 2:
            continue
        m = (BalancedBag().fit(X[tr], y[tr]) if kind == "bag"
             else make_clf(kind).fit(X[tr], y[tr]))
        oof[va] = m.predict_proba(X[va])[:, 1]
    return oof


def region_key(name: str) -> str:
    return "spine" if name.startswith("spine") else "femur"


def region_mask(data: pd.DataFrame, name: str) -> np.ndarray:
    key = region_key(name)
    return np.array([key in str(r) for r in data["region"].values], dtype=bool)


def build_sources(data, oof_v, real, kinds=("logreg", "lda", "bag")) -> dict:
    n = len(data)
    feats = {name: list(S.CRITERION_FEATURES.get(name, [])) for name in C.VIOLATIONS}
    out = {"cnn": np.full((n, len(C.VIOLATIONS)), np.nan)}
    for kind in kinds:
        out["geo_" + kind] = np.full((n, len(C.VIOLATIONS)), np.nan)
        out["fused_" + kind] = np.full((n, len(C.VIOLATIONS)), np.nan)
    for j, name in enumerate(C.VIOLATIONS):
        y = data["viol_" + name].values.astype(float)
        known = ~np.isnan(y) & region_mask(data, name) & real
        out["cnn"][known, j] = oof_v[known, j]
        cols = feats[name]
        if not cols or known.sum() == 0:
            continue
        X = data.loc[known, cols].fillna(0).values
        g = data.loc[known, "study_uid"].values
        for kind in kinds:
            out["geo_" + kind][known, j] = cv_score(X, y[known], g, kind)
            Xf = np.column_stack([oof_v[known, j], X])
            out["fused_" + kind][known, j] = cv_score(Xf, y[known], g, kind)
    return out


# --------------------------------------------------------------------------- #
# Агрегация по исследованию (идея #2)
# --------------------------------------------------------------------------- #
def aggregate(mat: np.ndarray, data, real, how: str) -> np.ndarray:
    if how == "none":
        return mat
    out = mat.copy()
    groups = data["study_uid"].values
    for j, name in enumerate(C.VIOLATIONS):
        lab = data["viol_" + name].values.astype(float)
        for g in np.unique(groups):
            idx = np.where((groups == g) & real & ~np.isnan(lab)
                           & np.isfinite(mat[:, j]))[0]
            if len(idx) == 0:
                continue
            v = mat[idx, j]
            if how == "mean":
                out[idx, j] = np.nanmean(v)
            elif how == "max":
                out[idx, j] = np.nanmax(v)
            elif how == "meanmax":
                out[idx, j] = 0.5 * (np.nanmean(v) + np.nanmax(v))
            elif how == "wmean":          # сильнее прижать к среднему (0.75/0.25)
                out[idx, j] = 0.75 * np.nanmean(v) + 0.25 * np.nanmax(v)
    return out


# --------------------------------------------------------------------------- #
# Прямая оптимизация порогов под macro-F1 организатора (идея #6)
# --------------------------------------------------------------------------- #
def _org_macro_on(data, probs, mask, thr) -> float:
    vals = []
    for lab, crits in C.ORG_LABELS.items():
        gt = np.zeros(int(mask.sum()))
        pr = np.zeros(int(mask.sum()))
        for k, c in enumerate(crits):
            v = np.nan_to_num(data["viol_" + c].values.astype(float)[mask], nan=0.0)
            p = (probs[mask, C.VIOLATION_IDX[c]] >= thr[c]).astype(float)
            gt = np.maximum(gt, v) if k else v
            pr = np.maximum(pr, p) if k else p
        if len(np.unique(gt)) > 1:
            vals.append(f1_score(gt, pr, zero_division=0))
    return float(np.mean(vals)) if vals else 0.0


def fit_org_thresholds(data, probs, mask, l2: float = 0.0) -> dict:
    """Coordinate ascent по 5 порогам; штраф l2*(t - blend)^2."""
    mask = np.asarray(mask, bool)
    grid = np.linspace(0.05, 0.95, 91)
    base = {}
    for i, name in enumerate(C.VIOLATIONS):
        y = data["viol_" + name].values.astype(float)[mask]
        p = probs[mask, i]
        ok = ~np.isnan(y) & np.isfinite(p)
        base[name] = (blend_threshold(y[ok], p[ok])
                      if ok.sum() and len(np.unique(y[ok])) > 1 else 0.5)
    thr = dict(base)

    def obj(t):
        pen = l2 * sum((t[c] - base[c]) ** 2 for c in C.VIOLATIONS)
        return _org_macro_on(data, probs, mask, t) - pen

    best = obj(thr)
    for _ in range(3):
        improved = False
        for name in C.VIOLATIONS:
            keep, best_t = thr[name], thr[name]
            for t in grid:
                thr[name] = float(t)
                v = obj(thr)
                if v > best + 1e-9:
                    best, best_t, improved = v, float(t), True
            thr[name] = best_t
        if not improved:
            break
    return thr


# --------------------------------------------------------------------------- #
# Оценка
# --------------------------------------------------------------------------- #
def _thresholds(data, thr_source, mask, mode, by_crit=None):
    if mode == "org":
        return fit_org_thresholds(data, thr_source, mask)
    t = pick_criterion_thresholds(data, thr_source, mask, mode=mode,
                                  by_criterion=by_crit)
    return {k: v["threshold"] for k, v in t.items()}


def evaluate(data, real, sel, thr_source, mode, seeds, by_crit=None):
    idx = np.where(real)[0]
    g = data["study_uid"].values[idx]
    strat = data["region"].values[idx]
    macros, per_all = [], {k: [] for k in ORG_LABEL_NAMES}
    q_all = {k: [] for k in ("ba", "mf1", "auc")}
    for seed in seeds:
        pl = {k: [] for k in ORG_LABEL_NAMES}
        q = {k: [] for k in ("ba", "mf1", "auc")}
        sp = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr_i, va_i in sp.split(idx, strat, groups=g):
            tr, va = idx[tr_i], idx[va_i]
            m = np.zeros(len(data), bool)
            m[tr] = True
            thr = _thresholds(data, thr_source, m, mode, by_crit)
            fired = {c: (sel[va, C.VIOLATION_IDX[c]] >= thr[c]).astype(float)
                     for c in C.VIOLATIONS}
            for lab, crits in C.ORG_LABELS.items():
                gt = np.zeros(len(va))
                pr = np.zeros(len(va))
                for k, c in enumerate(crits):
                    v = np.nan_to_num(data["viol_" + c].values.astype(float)[va],
                                      nan=0.0)
                    gt = np.maximum(gt, v) if k else v
                    pr = np.maximum(pr, fired[c]) if k else fired[c]
                pl[lab].append(f1_score(gt, pr, zero_division=0)
                               if len(np.unique(gt)) > 1 else np.nan)
            yq = data["quality"].values.astype(float)[va]
            regions = data["region"].values[va]
            qprob = np.full(len(va), np.nan)
            qcl = np.full(len(va), np.nan)
            for ii, row in enumerate(va):
                crit = C.REGION_CRITERIA[regions[ii]]
                probs = [sel[row, C.VIOLATION_IDX[c]] for c in crit]
                fin = [p for p in probs if np.isfinite(p)]
                qprob[ii] = max(fin) if fin else np.nan
                qcl[ii] = float(any(probs[k] >= thr[crit[k]]
                                    for k in range(len(crit))))
            ok = ~np.isnan(yq) & np.isfinite(qcl)
            if ok.sum() and len(np.unique(yq[ok])) > 1:
                q["ba"].append(balanced_accuracy_score(yq[ok], qcl[ok].astype(int)))
                q["mf1"].append(f1_score(yq[ok], qcl[ok].astype(int),
                                         average="macro", zero_division=0))
                q["auc"].append(roc_auc_score(yq[ok], qprob[ok]))
        macros.append(np.nanmean([np.nanmean(pl[k]) for k in ORG_LABEL_NAMES]))
        for k in ORG_LABEL_NAMES:
            per_all[k].append(np.nanmean(pl[k]))
        for k in ("ba", "mf1", "auc"):
            q_all[k].append(np.nanmean(q[k]))
    return (float(np.mean(macros)), float(np.std(macros)),
            {k: float(np.nanmean(v)) for k, v in per_all.items()},
            {k: float(np.nanmean(v)) for k, v in q_all.items()})


def select_scores(sources, data, real, src_map, agg):
    sel = np.full((len(data), len(C.VIOLATIONS)), np.nan)
    for j, name in enumerate(C.VIOLATIONS):
        sel[:, j] = aggregate(sources[src_map[name]], data, real, agg)[:, j]
    return sel


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--what", default="all",
                    choices=["all", "base", "agg", "clf", "org", "roi"])
    args = ap.parse_args()
    seeds = list(range(args.seeds))

    data = load_features(load_manifest(C.MANIFEST_CSV))
    oof_v = np.load(C.OOF_VIOLATION_NPY)
    tr_v = np.load(os.path.join(C.ARTIFACTS_DIR, "train_violation.npy"))
    real = real_mask(data)
    log("реальных снимков: %d   сидов: %d" % (int(real.sum()), args.seeds))

    sources = build_sources(data, oof_v, real)
    log("\n=== AUC источников по критериям (реальные) ===")
    for j, name in enumerate(C.VIOLATIONS):
        y = data["viol_" + name].values.astype(float)
        ok = ~np.isnan(y) & real & region_mask(data, name)
        line = "  %-20s n=%3d поз=%2d " % (name, int(ok.sum()), int(np.nansum(y[ok])))
        for s in sources:
            okk = ok & np.isfinite(sources[s][:, j])
            if len(np.unique(y[okk].astype(int))) > 1:
                line += "%s=%.3f " % (s.replace("_logreg", ""),
                                      roc_auc_score(y[okk].astype(int), sources[s][okk, j]))
        log(line)

    # карта источников: для каждого критерия — лучший по AUC (oracle-подобно, как база)
    auc_map = {}
    for j, name in enumerate(C.VIOLATIONS):
        y = data["viol_" + name].values.astype(float)
        ok = ~np.isnan(y) & real & region_mask(data, name)
        best_s, best_a = "cnn", -1.0
        for s in sources:
            okk = ok & np.isfinite(sources[s][:, j])
            if len(np.unique(y[okk].astype(int))) < 2:
                continue
            a = roc_auc_score(y[okk].astype(int), sources[s][okk, j])
            if a > best_a:
                best_a, best_s = a, s
        auc_map[name] = best_s
    log("  карта по AUC: %s" % auc_map)

    base_map = {name: "fused_logreg" for name in C.VIOLATIONS}
    base_map["femur_roi"] = "cnn"

    def run(title, src_map, mode, agg="none", thr_scale="cnn", l2=0.0):
        sel = select_scores(sources, data, real, src_map, agg)
        thr_src = (tr_v if thr_scale == "cnn"
                   else select_scores(sources, data, real, src_map, agg))
        if mode == "org":
            # org-порог всегда на выбранной шкале
            macro, sd, per, q = evaluate(data, real, sel, sel, mode, seeds)
        else:
            macro, sd, per, q = evaluate(data, real, sel, thr_src, mode, seeds)
        log("  %-46s macro=%.3f±%.3f | %s | BA=%.3f qF1=%.3f qAUC=%.3f"
            % (title, macro, sd,
               " ".join("%s=%.3f" % (k, v) for k, v in per.items()),
               q["ba"], q["mf1"], q["auc"]))
        return macro

    log("\n=== БАЗА и масштаб порога ===")
    run("base fused(roi=cnn) mapped cnn-scale", base_map, "mapped", "none", "cnn")
    run("base fused(roi=cnn) blend cnn-scale", base_map, "blend", "none", "cnn")
    run("base fused(roi=cnn) mapped clean-scale", base_map, "mapped", "none", "clean")
    run("base fused(roi=cnn) blend clean-scale", base_map, "blend", "none", "clean")
    log("  --- карта источников по AUC (выбор на всех данных, для ориентира) ---")
    run("карта AUC mapped clean-scale", auc_map, "mapped", "none", "clean")
    run("карта AUC blend clean-scale", auc_map, "blend", "none", "clean")
    geo_map = {n: ("geo_lda" if n in ("spine_positioning", "femur_roi")
                   else "geo_logreg") for n in C.VIOLATIONS}
    run("карта geo mapped clean-scale", geo_map, "mapped", "none", "clean")

    if args.what in ("all", "clf"):
        log("\n=== #5/#7 источники: LDA-shrinkage и balanced bagging ===")
        for s in ("geo_lda", "fused_lda", "geo_bag", "fused_bag"):
            sm = {n: s for n in C.VIOLATIONS}
            sm["femur_roi"] = "cnn"
            run("все=%s (roi=cnn) mapped cnn-scale" % s, sm, "mapped", "none", "cnn")

    if args.what in ("all", "roi"):
        log("\n=== femur_roi: источник x масштаб порога (остальные=fused_logreg) ===")
        for src in ("cnn", "geo_logreg", "geo_lda", "geo_bag",
                    "fused_logreg", "fused_bag"):
            for scale in ("cnn", "clean"):
                sm = dict(base_map)
                sm["femur_roi"] = src
                run("roi=%-12s scale=%s" % (src, scale), sm, "mapped",
                    "none", scale)

    if args.what in ("all", "agg"):
        log("\n=== #2 агрегация по исследованию (fused_logreg, mapped cnn-scale) ===")
        for how in ("none", "mean", "max", "meanmax", "wmean"):
            run("agg=%s" % how, base_map, "mapped", how, "cnn")

    if args.what in ("all", "org"):
        log("\n=== #6 прямая оптимизация порогов под macro-F1 организатора ===")
        for l2 in (0.0, 0.02, 0.05):
            run("org-thr l2=%.2f (clean-scale)" % l2, base_map, "org", "none", "clean")

    flush()
    log("\n[astra2] сводка: %s" % OUT_TXT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
