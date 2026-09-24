# -*- coding: utf-8 -*-
"""ФИНАЛЬНЫЕ метрики в формате ТЗ: Balanced Accuracy, Macro-F1, ROC-AUC (95% CI).

Считает по честной схеме (StratifiedGroupKFold по study_uid, порог по train-части,
метрики по реальным снимкам) для выбранной карты источников CNN/геометрия.

Запуск:
    python scripts/final_report.py --cnn out/cnn_resnet18,out/cnn_resnet34
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dxa_qc"))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import exp_fusion as X  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402


def bootstrap_ci_by_study(y, p, groups, metric="auc", n=2000, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n):
        pick = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq))])
        yy, pp = y[pick], p[pick]
        if len(np.unique(yy)) < 2:
            continue
        if metric == "auc":
            vals.append(roc_auc_score(yy, pp))
        elif metric == "ba":
            vals.append(balanced_accuracy_score(yy, pp))
        elif metric == "macro_f1":
            vals.append(f1_score(yy, pp, average="macro", zero_division=0))
        else:
            vals.append(f1_score(yy, pp, zero_division=0))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def select_map_on_train(data, real, S, tr):
    """Карта источников по критерию, выбранная ТОЛЬКО по train-части (train-AUC)."""
    source_map = {}
    for j, name in enumerate(C.VIOLATIONS):
        key = "spine" if name.startswith("spine") else "femur"
        rmask = X.region_mask(data, key)
        y = data["viol_" + name].values.astype(float)
        trc = tr[rmask[tr] & ~np.isnan(y[tr])]
        best_src, best_auc = "cnn", -1.0
        for s in X.SOURCES:
            p = S[s][:, j][trc]
            ok = ~np.isnan(p)
            if ok.sum() < 3 or len(np.unique(y[trc][ok].astype(int))) < 2:
                continue
            a = roc_auc_score(y[trc][ok].astype(int), p[ok])
            if a > best_auc:
                best_auc, best_src = a, s
        source_map[name] = best_src
    return source_map


def run_once(data, real, S, source_map, seed, thr_mode="blend"):
    idx = np.where(real)[0]
    g = data["study_uid"].values[idx]
    strat = data["region"].values[idx]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fired = np.zeros((len(data), len(C.VIOLATIONS)), dtype=int)
    oof_quality = np.full(len(data), np.nan)
    for tr_i, va_i in splitter.split(idx, strat, groups=g):
        tr, va = idx[tr_i], idx[va_i]
        sm = source_map if source_map is not None else select_map_on_train(data, real, S, tr)
        f = X.fired_from(data, real, S, sm, tr, va, thr_mode)
        fired[va] = f[va]
        oof_quality[va] = (f[va].sum(axis=1) > 0).astype(float)
    return fired, oof_quality


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn", default="out/cnn_resnet18")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--thr", default="blend",
                    choices=["prior", "f1", "blend", "kprior", "kf1", "kblend"],
                    help="режим порога (blend — валидированный дефолт)")
    ap.add_argument("--map", default="nested", choices=["nested", "auc", "team"],
                    help="отбор источников: nested (внутри train, честно) | auc | team")
    args = ap.parse_args()

    manifest = ds.load_manifest(C.MANIFEST_CSV)
    data = fl.load_features(manifest)
    real = np.asarray(ds.real_mask(data))
    dirs = [d.strip() for d in args.cnn.split(",") if d.strip()]
    oofs = [np.load(os.path.join(d, "oof_violation.npy")) for d in dirs]
    cnn_oof = np.nanmean(np.stack(oofs), axis=0)
    print(f"[final] CNN-ансамбль: {dirs}")

    S = X.build_scores(data, real, cnn_oof)

    # карта источников: по AUC каждого критерия (устойчива, без подглядывания в macro)
    auc_map = {}
    for j, name in enumerate(C.VIOLATIONS):
        y = data["viol_" + name].values.astype(float)
        ok = ~np.isnan(y) & real
        best_src, best_auc = "cnn", -1.0
        for s in X.SOURCES:
            okk = ok & ~np.isnan(S[s][:, j])
            if len(np.unique(y[okk].astype(int))) < 2:
                continue
            a = roc_auc_score(y[okk].astype(int), S[s][okk, j])
            if a > best_auc:
                best_auc, best_src = a, s
        auc_map[name] = best_src
        print(f"  {name:20} -> {best_src}  (AUC={best_auc:.3f})")

    team_map = {"spine_positioning": "cnn", "spine_axis": "fused",
                "spine_artifacts": "cnn", "femur_positioning": "cnn",
                "femur_roi": "cnn"}
    if args.map == "nested":
        fixed_map = None
        print("  отбор карты: ВНУТРИ train-фолда (nested, честно)")
    elif args.map == "auc":
        fixed_map = auc_map
    else:
        fixed_map = team_map

    # --- организаторская macro-F1 по 4 меткам, усреднение по сидам ---
    per_label_seed = {k: [] for k in X.ORG_LABELS}
    macro_seed, qba_seed, qf1_seed, qauc_seed = [], [], [], []
    q_oof_accum = np.zeros((len(data), args.seeds))
    # накопление бинарных решений организатора по меткам (для CI macro-F1)
    org_acc = {lab: np.zeros((len(data), args.seeds), dtype=float) for lab in X.ORG_LABELS}
    yq = data["quality"].values.astype(float)
    yb = np.nan_to_num(yq, nan=0.0).astype(int)
    for seed in range(args.seeds):
        fired, oof_q = run_once(data, real, S, fixed_map, seed, args.thr)
        for lab, crits in X.ORG_LABELS.items():
            gt = np.zeros(len(data), dtype=int)
            pr = np.zeros(len(data), dtype=int)
            for name in crits:
                gt |= np.nan_to_num(data["viol_" + name].values.astype(float), nan=0.0).astype(int)
                pr |= fired[:, C.VIOLATION_IDX[name]]
            per_label_seed[lab].append(f1_score(gt[real], pr[real], zero_division=0))
            org_acc[lab][:, seed] = pr
        macro_seed.append(np.mean([per_label_seed[k][-1] for k in X.ORG_LABELS]))
        pq = (fired.sum(axis=1) > 0).astype(int)
        qba_seed.append(balanced_accuracy_score(yb[real], pq[real]))
        qf1_seed.append(f1_score(yb[real], pq[real], average="macro", zero_division=0))
        qauc_seed.append(roc_auc_score(yb[real], pq[real]))
        q_oof_accum[:, seed] = oof_q

    # --- 95% CI по исследованиям (на усреднённом по сидам решении) ---
    pq_mean = (q_oof_accum.mean(axis=1) > 0.5).astype(int)
    gg = data["study_uid"].values[real]
    yy = yb[real]
    pp = pq_mean[real]
    ba_ci = bootstrap_ci_by_study(yy, pp, gg, "ba")
    f1_ci = bootstrap_ci_by_study(yy, pp, gg, "macro_f1")

    # CI macro-F1 организатора: бутстрэп по исследованиям, решения усреднены по сидам
    rng = np.random.default_rng(0)
    uniq = np.unique(gg)
    idx_by_g = {g: np.where(gg == g)[0] for g in uniq}
    lab_names = list(X.ORG_LABELS)
    # gt и pr по меткам на реальных снимках
    org_gt = {}
    for lab, crits in X.ORG_LABELS.items():
        gt = np.zeros(len(data), dtype=int)
        for name in crits:
            gt |= np.nan_to_num(data["viol_" + name].values.astype(float), nan=0.0).astype(int)
        org_gt[lab] = gt[real]
    org_pr = {lab: (org_acc[lab].mean(axis=1) > 0.5).astype(int)[real] for lab in lab_names}
    macro_vals = []
    for _ in range(2000):
        pick = np.concatenate([idx_by_g[g] for g in rng.choice(uniq, len(uniq))])
        f1s = [f1_score(org_gt[lab][pick], org_pr[lab][pick], zero_division=0)
               for lab in lab_names]
        macro_vals.append(np.mean(f1s))
    macro_ci = (float(np.percentile(macro_vals, 2.5)),
                float(np.percentile(macro_vals, 97.5)))

    print("\n" + "=" * 78)
    print("ИТОГОВЫЕ МЕТРИКИ (честно, StratifiedGroupKFold по study_uid, порог по train)")
    print(f"режим порога: {args.thr}")
    print("=" * 78)
    print(f"{'Метрика':34} {'значение':>10} {'±std':>8}")
    print(f"{'Macro-F1 (4 метки организатора)':34} {np.mean(macro_seed):10.3f} {np.std(macro_seed):8.3f}")
    print(f"{'Balanced Accuracy (quality)':34} {np.mean(qba_seed):10.3f} {np.std(qba_seed):8.3f}")
    print(f"{'Macro-F1 quality_class':34} {np.mean(qf1_seed):10.3f} {np.std(qf1_seed):8.3f}")
    print(f"{'ROC-AUC quality_class':34} {np.mean(qauc_seed):10.3f} {np.std(qauc_seed):8.3f}")
    print("\nпо меткам организатора:")
    for lab in X.ORG_LABELS:
        print(f"  {lab:34} F1={np.mean(per_label_seed[lab]):.3f} ± {np.std(per_label_seed[lab]):.3f}")

    print(f"\n95% CI (бутстрэп по исследованиям, {args.seeds} сидов усреднены):")
    print(f"  Macro-F1 (орг.) {np.mean(macro_seed):.3f}  [{macro_ci[0]:.3f}; {macro_ci[1]:.3f}]")
    print(f"  BA   {np.mean(qba_seed):.3f}  [{ba_ci[0]:.3f}; {ba_ci[1]:.3f}]")
    print(f"  F1 (macro quality) {np.mean(qf1_seed):.3f}  [{f1_ci[0]:.3f}; {f1_ci[1]:.3f}]")
    print(f"\nкарта источников (для справки, по AUC на всём наборе): {auc_map}")
    print(f"режим отбора карты: {args.map}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
