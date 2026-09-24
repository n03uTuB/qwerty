# -*- coding: utf-8 -*-
"""Фьюжн CNN + геометрия и честный отбор источника по критерию.

Цель — поднять ИТОГОВЫЕ метрики организатора:
  * macro-F1 по 4 уникальным меткам (укладка/ось/предметы/ROI);
  * метрики quality_class (balanced accuracy, macro-F1, ROC-AUC).

Схема честная:
  * разбиение GroupKFold по study_uid (пациент не пересекает фолды);
  * порог классификации — по обучающей части фолда (prior);
  * отбор источника по критерию делается ВНУТРИ обучающей части (nested),
    чтобы не подглядывать в валидацию;
  * метрики — только по реальным снимкам.

Запуск:
    python scripts/exp_fusion.py [--cnn out/cnn_resnet18] [--metric auc|f1]
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, f1_score, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dxa_qc"))
sys.path.insert(0, os.path.join(HERE, ".."))

from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402

ORG_LABELS = {
    "укладка": ["spine_positioning", "femur_positioning"],
    "ось": ["spine_axis"],
    "предметы": ["spine_artifacts"],
    "ROI": ["femur_roi"],
}

GEO_SETS = {
    "spine_positioning": ["spine_iliac_signal", "spine_bottom_cut"],
    "spine_axis": ["spine_midline_angle"],
    "spine_artifacts": ["spine_ribs_signal", "spine_vertebra_peaks", "spine_midline_residual"],
    "femur_positioning": ["femur_trochanter_bulge", "femur_troch_area_mm2", "femur_neck_width_mm"],
    "femur_roi": ["femur_height_cm", "femur_bone_ratio", "femur_margin_min_cm"],
}
GEO_BIG = {
    "spine_positioning": ["spine_axis_angle", "spine_midline_angle", "spine_midline_residual",
                          "spine_iliac_signal", "spine_ribs_signal", "spine_vertebra_peaks",
                          "spine_top_cut", "spine_bottom_cut", "spine_margin_bottom_cm",
                          "spine_height_cm", "spine_bone_ratio"],
    "spine_axis": ["spine_axis_angle", "spine_midline_angle", "spine_midline_residual",
                   "spine_bone_ratio", "vertical_symmetry"],
    "spine_artifacts": ["spine_ribs_signal", "spine_vertebra_peaks", "spine_midline_residual",
                        "spine_bone_ratio", "bright_area_ratio", "vertical_symmetry",
                        "bone_eccentricity"],
    "femur_positioning": ["femur_axis_angle", "femur_bone_ratio", "femur_width_cm",
                          "femur_margin_min_cm", "femur_trochanter_bulge", "femur_troch_area_mm2",
                          "femur_neck_width_mm", "femur_head_diameter_mm", "femur_neck_to_head",
                          "femur_shaft_deg", "bone_eccentricity"],
    "femur_roi": ["femur_height_cm", "femur_width_cm", "femur_bone_ratio", "femur_margin_min_cm",
                  "femur_margin_top_cm", "femur_margin_bottom_cm", "bright_area_ratio"],
}
SOURCES = ("cnn", "geo", "geobig", "fused", "fusedbig")


def clf():
    return Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))])


def region_mask(data, key):
    return np.array([key in str(r) for r in data["region"].values], dtype=bool)


def prior_threshold(y, p):
    prev = float(np.mean(y)) if len(y) else 0.0
    if prev <= 0 or prev >= 1 or len(p) == 0:
        return 0.5
    k = max(1, int(round(prev * len(p))))
    return float(np.sort(p)[::-1][min(k, len(p)) - 1])


def best_f1_threshold(y, p):
    """Порог, максимизирующий F1 на переданных скорах (обычно train-часть)."""
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def pick_threshold(y, p, mode="prior"):
    """Режим порога: prior (топ-k по доле) | f1 (опт. F1) | blend (полусумма)."""
    if mode == "f1":
        return best_f1_threshold(y, p)
    if mode == "blend":
        return 0.5 * (prior_threshold(y, p) + best_f1_threshold(y, p))
    return prior_threshold(y, p)


def best_f1_k(y, p):
    """Число верхних по скору, максимизирующее F1 на train (ранговое правило)."""
    order = np.argsort(p)[::-1]
    y_sorted = y[order]
    best_k, best_f1 = 1, -1.0
    for k in range(1, len(y_sorted) + 1):
        f1 = f1_score(y_sorted, np.arange(len(y_sorted)) < k, zero_division=0)
        if f1 > best_f1:
            best_f1, best_k = f1, k
    return int(best_k)


def prior_k(y):
    return max(1, int(round(float(np.mean(y)) * len(y))))


def k_threshold(p, k):
    k = min(max(int(k), 1), len(p))
    return float(np.sort(p)[::-1][k - 1])


def pick_k(y, p, mode="prior"):
    """Ранговое (top-k) правило: k подбирается на train, применяется на val.

    kprior — k по доле; kf1 — k по максимуму F1; kblend — полусумма обоих.
    """
    if mode == "kf1":
        return best_f1_k(y, p)
    if mode == "kblend":
        return int(round(0.5 * (prior_k(y) + best_f1_k(y, p))))
    return prior_k(y)


def cv_score(X, y, groups, n_splits=5):
    oof = np.full(len(y), np.nan)
    for tr, va in GroupKFold(n_splits=n_splits).split(X, groups=groups):
        if len(np.unique(y[tr])) < 2:
            continue
        m = clf().fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    return oof


def build_scores(data, real, cnn_oof):
    n = len(data)
    S = {s: np.full((n, len(C.VIOLATIONS)), np.nan) for s in SOURCES}
    for j, name in enumerate(C.VIOLATIONS):
        key = "spine" if name.startswith("spine") else "femur"
        rmask = region_mask(data, key)
        y = data["viol_" + name].values.astype(float)
        known = ~np.isnan(y) & rmask & real
        if known.sum() == 0:
            continue
        S["cnn"][known, j] = cnn_oof[known, j]
        for src, sets in (("geo", GEO_SETS), ("geobig", GEO_BIG)):
            cols = [c for c in sets[name] if c in data.columns]
            X = data.loc[known, cols].fillna(0).values
            S[src][known, j] = cv_score(X, y[known].astype(int),
                                        data.loc[known, "study_uid"].values)
        for src, sets in (("fused", GEO_SETS), ("fusedbig", GEO_BIG)):
            cols = [c for c in sets[name] if c in data.columns]
            X = np.column_stack([cnn_oof[known, j], data.loc[known, cols].fillna(0).values])
            S[src][known, j] = cv_score(X, y[known].astype(int),
                                        data.loc[known, "study_uid"].values)
    return S


def fired_from(data, real, S, source_map, tr, va, thr_mode="prior"):
    fired = np.zeros((len(data), len(C.VIOLATIONS)), dtype=int)
    for j, name in enumerate(C.VIOLATIONS):
        key = "spine" if name.startswith("spine") else "femur"
        rmask = region_mask(data, key)
        y = data["viol_" + name].values.astype(float)
        p = S[source_map[name]][:, j]
        trc = tr[rmask[tr] & ~np.isnan(y[tr]) & ~np.isnan(p[tr])]
        if len(trc) < 2 or len(np.unique(y[trc].astype(int))) < 2:
            continue
        ytr, ptr = y[trc].astype(int), p[trc]
        va_ok = va[~np.isnan(p[va])]
        if thr_mode in ("kprior", "kf1", "kblend"):
            k = pick_k(ytr, ptr, thr_mode)
            va_ok = va_ok[np.argsort(p[va_ok])[::-1][:k]]
            fired[va_ok, j] = 1
        elif thr_mode == "prank":
            # prior по РАНГУ val-скоров (калибровочно-устойчиво)
            k = max(1, int(round(float(np.mean(ytr)) * len(va_ok))))
            va_ok = va_ok[np.argsort(p[va_ok])[::-1][:k]]
            fired[va_ok, j] = 1
        else:
            thr = pick_threshold(ytr, ptr, thr_mode)
            fired[va_ok[p[va_ok] >= thr], j] = 1
    return fired


def org_scores(data, fired, va):
    out = {}
    for lab, crits in ORG_LABELS.items():
        gt = np.zeros(len(data), dtype=int)
        pr = np.zeros(len(data), dtype=int)
        for name in crits:
            v = data["viol_" + name].values.astype(float)
            gt |= np.nan_to_num(v, nan=0.0).astype(int)
            pr |= fired[:, C.VIOLATION_IDX[name]]
        out[lab] = f1_score(gt[va], pr[va], zero_division=0)
    return out


def quality_scores(data, real, fired, va):
    yq = data["quality"].values.astype(float)
    yb = np.nan_to_num(yq, nan=0.0).astype(int)
    pq = (fired.sum(axis=1) > 0).astype(int)
    yy, pp = yb[va], pq[va]
    ba = balanced_accuracy_score(yy, pp) if len(np.unique(yy)) > 1 else float("nan")
    mf1 = f1_score(yy, pp, average="macro", zero_division=0)
    auc = roc_auc_score(yy, pp) if len(np.unique(yy)) > 1 else float("nan")
    return dict(ba=ba, mf1=mf1, auc=auc)


def eval_fixed(data, real, S, source_map):
    idx = np.where(real)[0]
    g = data["study_uid"].values[idx]
    per_label = {k: [] for k in ORG_LABELS}
    q = {k: [] for k in ("ba", "mf1", "auc")}
    for tr_i, va_i in GroupKFold(n_splits=5).split(idx, groups=g):
        tr, va = idx[tr_i], idx[va_i]
        fired = fired_from(data, real, S, source_map, tr, va)
        lab = org_scores(data, fired, va)
        for k in ORG_LABELS:
            per_label[k].append(lab[k])
        qq = quality_scores(data, real, fired, va)
        for k in q:
            q[k].append(qq[k])
    macro = float(np.mean([np.mean(v) for v in per_label.values()]))
    return macro, {k: float(np.mean(v)) for k, v in per_label.items()}, \
        {k: float(np.nanmean(v)) for k, v in q.items()}


def eval_fixed_seeds(data, real, S, source_map, seeds=range(10), thr_mode="prior"):
    """macro-F1 и quality-метрики, усреднённые по нескольким разбиениям (mean/std)."""
    from sklearn.model_selection import StratifiedGroupKFold
    macros, per_all, q_all = [], {k: [] for k in ORG_LABELS}, {k: [] for k in ("ba", "mf1", "auc")}
    idx = np.where(real)[0]
    g = data["study_uid"].values[idx]
    strat = data["region"].values[idx]
    for seed in seeds:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        per_label = {k: [] for k in ORG_LABELS}
        q = {k: [] for k in ("ba", "mf1", "auc")}
        for tr_i, va_i in splitter.split(idx, strat, groups=g):
            tr, va = idx[tr_i], idx[va_i]
            fired = fired_from(data, real, S, source_map, tr, va, thr_mode)
            lab = org_scores(data, fired, va)
            for k in ORG_LABELS:
                per_label[k].append(lab[k])
            qq = quality_scores(data, real, fired, va)
            for k in q:
                q[k].append(qq[k])
        macros.append(np.mean([np.mean(v) for v in per_label.values()]))
        for k in ORG_LABELS:
            per_all[k].append(np.mean(per_label[k]))
        for k in q:
            q_all[k].append(np.nanmean(q[k]))
    return (float(np.mean(macros)), float(np.std(macros)),
            {k: float(np.mean(v)) for k, v in per_all.items()},
            {k: float(np.mean(v)) for k, v in q_all.items()})


def eval_nested(data, real, S, metric="auc"):
    idx = np.where(real)[0]
    g = data["study_uid"].values[idx]
    per_label = {k: [] for k in ORG_LABELS}
    q = {k: [] for k in ("ba", "mf1", "auc")}
    chosen = {name: {} for name in C.VIOLATIONS}
    for tr_i, va_i in GroupKFold(n_splits=5).split(idx, groups=g):
        tr, va = idx[tr_i], idx[va_i]
        source_map = {}
        for j, name in enumerate(C.VIOLATIONS):
            key = "spine" if name.startswith("spine") else "femur"
            rmask = region_mask(data, key)
            y = data["viol_" + name].values.astype(float)
            best_src, best_val = "cnn", -1.0
            for s in SOURCES:
                p = S[s][:, j]
                trc = tr[rmask[tr] & ~np.isnan(y[tr]) & ~np.isnan(p[tr])]
                if len(trc) < 3 or len(np.unique(y[trc].astype(int))) < 2:
                    continue
                if metric == "auc":
                    val = roc_auc_score(y[trc].astype(int), p[trc])
                else:
                    val = f1_score(y[trc].astype(int),
                                   (p[trc] >= prior_threshold(y[trc].astype(int), p[trc])
                                    ).astype(int), zero_division=0)
                if val > best_val:
                    best_val, best_src = val, s
            source_map[name] = best_src
            chosen[name][best_src] = chosen[name].get(best_src, 0) + 1
        fired = fired_from(data, real, S, source_map, tr, va)
        lab = org_scores(data, fired, va)
        for k in ORG_LABELS:
            per_label[k].append(lab[k])
        qq = quality_scores(data, real, fired, va)
        for k in q:
            q[k].append(qq[k])
    macro = float(np.mean([np.mean(v) for v in per_label.values()]))
    return macro, {k: float(np.mean(v)) for k, v in per_label.items()}, \
        {k: float(np.nanmean(v)) for k, v in q.items()}, chosen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn", default="out/cnn_resnet18",
                    help="каталог(и) с oof_violation.npy через запятую — усредняются")
    ap.add_argument("--metric", default="auc", choices=["auc", "f1"])
    args = ap.parse_args()

    manifest = ds.load_manifest(C.MANIFEST_CSV)
    data = fl.load_features(manifest)
    real = np.asarray(ds.real_mask(data))
    dirs = [d.strip() for d in args.cnn.split(",") if d.strip()]
    oofs = [np.load(os.path.join(d, "oof_violation.npy")) for d in dirs]
    cnn_oof = np.nanmean(np.stack(oofs), axis=0)
    print(f"[fusion] строк={len(data)}  реальных={int(real.sum())}  CNN={dirs}")

    S = build_scores(data, real, cnn_oof)

    print("\n=== AUC по критериям (реальные снимки) ===")
    for j, name in enumerate(C.VIOLATIONS):
        y = data["viol_" + name].values.astype(float)
        ok = ~np.isnan(y) & real & ~np.isnan(S["cnn"][:, j])
        line = f"  {name:20} n={int(ok.sum()):3d} поз={int(np.nansum(y[ok])):2d}  "
        for s in SOURCES:
            okk = ok & ~np.isnan(S[s][:, j])
            if len(np.unique(y[okk].astype(int))) < 2:
                line += f"{s}=n/a  "
            else:
                line += f"{s}={roc_auc_score(y[okk].astype(int), S[s][okk, j]):.3f}  "
        print(line)

    print("\n=== Один источник на все критерии ===")
    for s in ("cnn", "geo", "fused", "fusedbig"):
        macro, per, q = eval_fixed(data, real, S, {c: s for c in C.VIOLATIONS})
        print(f"  {s:9} macro-F1={macro:.3f}  " +
              " ".join(f"{k}={v:.3f}" for k, v in per.items()) +
              f"  | BA={q['ba']:.3f} qF1={q['mf1']:.3f} qAUC={q['auc']:.3f}")

    print("\n=== Полный перебор карт источников (oracle по macro-F1) ===")
    rows = []
    for combo in itertools.product(SOURCES, repeat=len(C.VIOLATIONS)):
        sm = dict(zip(C.VIOLATIONS, combo))
        macro, per, q = eval_fixed(data, real, S, sm)
        rows.append((macro, combo, per, q))
    rows.sort(key=lambda r: -r[0])
    for macro, combo, per, q in rows[:8]:
        print(f"  {macro:.3f}  {dict(zip(C.VIOLATIONS, combo))}")
    best_macro, best_combo, best_per, best_q = rows[0]
    print(f"  ЛУЧШЕЕ(oracle) macro-F1={best_macro:.3f}  " +
          " ".join(f"{k}={v:.3f}" for k, v in best_per.items()) +
          f"  | BA={best_q['ba']:.3f} qF1={best_q['mf1']:.3f} qAUC={best_q['auc']:.3f}")

    print("\n=== ЧЕСТНЫЙ nested-отбор источника (по train-части) ===")
    macro, per, q, chosen = eval_nested(data, real, S, metric=args.metric)
    print(f"  nested macro-F1={macro:.3f}  " +
          " ".join(f"{k}={v:.3f}" for k, v in per.items()) +
          f"  | BA={q['ba']:.3f} qF1={q['mf1']:.3f} qAUC={q['auc']:.3f}")
    print("  выбор источника:", {k: v for k, v in chosen.items()})

    # --- карта, выбранная по AUC каждого критерия (слабее переобучается, чем по macro) ---
    auc_map = {}
    for j, name in enumerate(C.VIOLATIONS):
        y = data["viol_" + name].values.astype(float)
        ok = ~np.isnan(y) & real
        best_src, best_auc = "cnn", -1.0
        for s in SOURCES:
            okk = ok & ~np.isnan(S[s][:, j])
            if len(np.unique(y[okk].astype(int))) < 2:
                continue
            a = roc_auc_score(y[okk].astype(int), S[s][okk, j])
            if a > best_auc:
                best_auc, best_src = a, s
        auc_map[name] = best_src
    print("\n  карта по AUC критерия:", auc_map)

    # --- сравнение ключевых карт по 10 разбиениям (mean±std) ---
    maps = {
        "текущий config": {"spine_positioning": "cnn", "spine_axis": "fused",
                           "spine_artifacts": "cnn", "femur_positioning": "cnn",
                           "femur_roi": "cnn"},
        "база: fused везде": {c: "fused" for c in C.VIOLATIONS},
        "fusedbig везде": {c: "fusedbig" for c in C.VIOLATIONS},
        "geo везде": {c: "geo" for c in C.VIOLATIONS},
        "cnn везде": {c: "cnn" for c in C.VIOLATIONS},
        "карта по AUC": auc_map,
        "oracle": dict(zip(C.VIOLATIONS, best_combo)),
    }
    print("\n=== Сравнение карт: 10 разбиений, mean±std ===")
    for title, sm in maps.items():
        m, sd, per, q = eval_fixed_seeds(data, real, S, sm)
        print(f"  {title:20} macro-F1={m:.3f}±{sd:.3f}  " +
              " ".join(f"{k}={v:.3f}" for k, v in per.items()) +
              f"  | BA={q['ba']:.3f} qF1={q['mf1']:.3f} qAUC={q['auc']:.3f}")

    np.savez(os.path.join("out", "fusion_S.npz"),
             **{f"{s}": S[s] for s in SOURCES}, real=real)
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
