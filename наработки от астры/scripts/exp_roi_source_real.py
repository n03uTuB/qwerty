# -*- coding: utf-8 -*-
"""Честный замер источника скора для `femur_roi` на БОЕВОМ пайплайне.

Стенд `exp_astra2.py` показал seed-averaged выигрыш от смены источника ROI
(`cnn` -> `fused`/`geo`). Здесь то же проверяется на реальном контуре оценки
(`src.stack.build_scores` + честные функции `src.evaluate`).

Протоколы:
  * ``fold_id`` — исторический боевой: фолды из обучения CNN (`fold_id.npy`);
  * ``seeds``   — StratifiedGroupKFold(5) по study_uid x N сидов (устойчивее).

Порог выбирается ТОЛЬКО по train-части фолда. Два масштаба:
  * ``cnn``   — порог по train-пробам CNN (`train_violation.npy`, исторический);
  * ``clean`` — порог по OOF-скорам стекера на train-части (та же шкала, что val).

Печатает macro-F1 организатора, per-label F1, quality_class BA/macro-F1/AUC и
per-criterion honest F1. Плюс парный кластер-бутстрэп по исследованиям для
значимости дельты против базы.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from sklearn.metrics import (balanced_accuracy_score, f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from src import config as C          # noqa: E402
from src import stack as S           # noqa: E402
from src.data import load_manifest, real_mask   # noqa: E402
from src.features import load_features          # noqa: E402
from src.metrics import (ORG_LABEL_NAMES, org_label_truth,  # noqa: E402
                         pick_criterion_thresholds)

OUT_TXT = os.path.join(C.PROJECT_ROOT, "results", "roi_source_real.txt")
_LINES: list = []


def log(msg: str) -> None:
    print(msg)
    _LINES.append(msg)


# --------------------------------------------------------------------------- #
# Пороги и метрики одного разбиения
# --------------------------------------------------------------------------- #
def _thresholds(df, thr_src, mask, roi_mode=None):
    by_crit = dict(C.THRESHOLD_MODE_BY_CRITERION)
    if roi_mode:
        by_crit["femur_roi"] = roi_mode
    return pick_criterion_thresholds(df, thr_src, mask, mode="mapped",
                                     by_criterion=by_crit)


def _qc_on_val(df, oof_v, va, thr, real):
    regions = df["region"].values
    q_prob = np.full(len(df), np.nan)
    q_class = np.full(len(df), np.nan)
    for i in va:
        crit = C.REGION_CRITERIA[regions[i]]
        probs = [oof_v[i, C.VIOLATION_IDX[c]] for c in crit]
        fired = [p >= thr[c]["threshold"] for p, c in zip(probs, crit)]
        fin = [p for p in probs if np.isfinite(p)]
        q_prob[i] = max(fin) if fin else np.nan
        q_class[i] = float(any(fired))
    vq = np.isfinite(q_class) & real
    yq = df["quality"].values.astype(float)
    vq &= ~np.isnan(yq)
    if vq.sum() == 0 or len(np.unique(yq[vq])) < 2:
        return dict(ba=float("nan"), mf1=float("nan"), auc=float("nan"))
    pred = q_class[vq].astype(int)
    return dict(ba=balanced_accuracy_score(yq[vq], pred),
                mf1=f1_score(yq[vq], pred, average="macro", zero_division=0),
                auc=roc_auc_score(yq[vq], q_prob[vq]))


def _org_on_val(df, oof_v, va, thr):
    per = {}
    for name in ORG_LABEL_NAMES:
        truth = org_label_truth(df, name)
        members = [(oof_v[va, C.VIOLATION_IDX[c]] >= thr[c]["threshold"]
                    ).astype(float) for c in C.ORG_LABELS[name]]
        pred = np.max(np.vstack(members), axis=0)
        y = truth[va]
        m = np.isfinite(y)
        per[name] = (f1_score(y[m], pred[m], zero_division=0)
                     if m.sum() and len(np.unique(y[m])) > 1 else np.nan)
    return per


def _folds(df, real, protocol, seeds):
    if protocol == "fold_id":
        fold_id = np.load(os.path.join(C.ARTIFACTS_DIR, "fold_id.npy"))
        return [[np.where((fold_id == k) & real)[0]
                 for k in sorted(set(fold_id[fold_id >= 0].tolist()))]]
    idx = np.where(real)[0]
    g = df["study_uid"].values[idx]
    strat = df["region"].values[idx]
    out = []
    for seed in range(seeds):
        sp = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        out.append([idx[vi] for _, vi in sp.split(idx, strat, groups=g)])
    return out


def _pooled_foldid_protocol(df, stack_v, tr_v, real, scale, roi_mode):
    """Пул val-предсказаний по фолдам fold_id (как в src.evaluate)."""
    fold_id = np.load(os.path.join(C.ARTIFACTS_DIR, "fold_id.npy"))
    n = len(df)
    org_pred = {k: np.full(n, np.nan) for k in ORG_LABEL_NAMES}
    q_class = np.full(n, np.nan)
    q_prob = np.full(n, np.nan)
    regions = df["region"].values
    for k in sorted(set(fold_id[fold_id >= 0].tolist())):
        va = np.where((fold_id == k) & real)[0]
        tr = (fold_id != k) & real & np.isfinite(tr_v[:, 0])
        thr_src = tr_v if scale == "cnn" else stack_v
        thr = _thresholds(df, thr_src, tr, roi_mode)
        for name in ORG_LABEL_NAMES:
            members = [(stack_v[va, C.VIOLATION_IDX[c]] >= thr[c]["threshold"]
                        ).astype(float) for c in C.ORG_LABELS[name]]
            org_pred[name][va] = np.max(np.vstack(members), axis=0)
        for i in va:
            crit = C.REGION_CRITERIA[regions[i]]
            probs = [stack_v[i, C.VIOLATION_IDX[c]] for c in crit]
            fired = [p >= thr[c]["threshold"] for p, c in zip(probs, crit)]
            fin = [p for p in probs if np.isfinite(p)]
            q_prob[i] = max(fin) if fin else np.nan
            q_class[i] = float(any(fired))
    return org_pred, q_class, q_prob


def _f1_pooled(truth, pred):
    m = np.isfinite(pred) & np.isfinite(truth)
    if m.sum() == 0 or len(np.unique(truth[m])) < 2:
        return float("nan")
    return float(f1_score(truth[m], (pred[m] >= 0.5).astype(int),
                          zero_division=0))


def evaluate_protocol(df, stack_v, tr_v, real, protocol, scale, seeds,
                      roi_mode=None):
    if protocol == "fold_id":
        org_pred, q_class, q_prob = _pooled_foldid_protocol(
            df, stack_v, tr_v, real, scale, roi_mode)
        per = {k: _f1_pooled(org_label_truth(df, k), org_pred[k])
               for k in ORG_LABEL_NAMES}
        macro = float(np.nanmean(list(per.values())))
        yq = df["quality"].values.astype(float)
        vq = np.isfinite(q_class) & real & ~np.isnan(yq)
        q = dict(ba=balanced_accuracy_score(yq[vq], q_class[vq].astype(int)),
                 mf1=f1_score(yq[vq], q_class[vq].astype(int), average="macro",
                              zero_division=0),
                 auc=roc_auc_score(yq[vq], q_prob[vq]))
        per_seed = dict(macro=[macro], ba=[q["ba"]], mf1=[q["mf1"]],
                        auc=[q["auc"]])
        return macro, 0.0, per, q, per_seed

    macros, per_all = [], {k: [] for k in ORG_LABEL_NAMES}
    q_all = {k: [] for k in ("ba", "mf1", "auc")}
    for fold_list in _folds(df, real, protocol, seeds):
        pl = {k: [] for k in ORG_LABEL_NAMES}
        q = {k: [] for k in ("ba", "mf1", "auc")}
        for va in fold_list:
            m = np.zeros(len(df), bool)
            m[va] = True
            tr = real & ~m & np.isfinite(tr_v[:, 0])
            thr_src = tr_v if scale == "cnn" else stack_v
            thr = _thresholds(df, thr_src, tr, roi_mode)
            for k, v in _org_on_val(df, stack_v, va, thr).items():
                pl[k].append(v)
            qm = _qc_on_val(df, stack_v, va, thr, real)
            for k in q:
                q[k].append(qm[k])
        macros.append(np.nanmean([np.nanmean(pl[k]) for k in ORG_LABEL_NAMES]))
        for k in ORG_LABEL_NAMES:
            per_all[k].append(np.nanmean(pl[k]))
        for k in q:
            q_all[k].append(np.nanmean(q[k]))
    per_seed = dict(macro=list(macros),
                    ba=list(q_all["ba"]), mf1=list(q_all["mf1"]),
                    auc=list(q_all["auc"]))
    return (float(np.mean(macros)), float(np.std(macros)),
            {k: float(np.nanmean(v)) for k, v in per_all.items()},
            {k: float(np.nanmean(v)) for k, v in q_all.items()},
            per_seed)


# --------------------------------------------------------------------------- #
# Парный кластер-бутстрэп по исследованиям (пул val по фолдам fold_id)
# --------------------------------------------------------------------------- #
def _pooled(df, oof_v, tr_v, real, scale, roi_mode=None):
    fold_id = np.load(os.path.join(C.ARTIFACTS_DIR, "fold_id.npy"))
    n = len(df)
    org_pred = {k: np.full(n, np.nan) for k in ORG_LABEL_NAMES}
    q_class = np.full(n, np.nan)
    regions = df["region"].values
    for k in sorted(set(fold_id[fold_id >= 0].tolist())):
        va = np.where((fold_id == k) & real)[0]
        tr = (fold_id != k) & real & np.isfinite(tr_v[:, 0])
        thr_src = tr_v if scale == "cnn" else oof_v
        thr = _thresholds(df, thr_src, tr, roi_mode)
        for name in ORG_LABEL_NAMES:
            members = [(oof_v[va, C.VIOLATION_IDX[c]] >= thr[c]["threshold"]
                        ).astype(float) for c in C.ORG_LABELS[name]]
            org_pred[name][va] = np.max(np.vstack(members), axis=0)
        for i in va:
            crit = C.REGION_CRITERIA[regions[i]]
            probs = [oof_v[i, C.VIOLATION_IDX[c]] for c in crit]
            fired = [p >= thr[c]["threshold"] for p, c in zip(probs, crit)]
            q_class[i] = float(any(fired))
    return org_pred, q_class


def paired_bootstrap(df, oof_a, oof_b, real, scale, roi_mode=None,
                     n_boot=2000, seed=0):
    truth = {k: org_label_truth(df, k) for k in ORG_LABEL_NAMES}
    pa, qa = _pooled(df, oof_a, tr_v_g, real, scale, roi_mode)
    pb, qb = _pooled(df, oof_b, tr_v_g, real, scale, roi_mode)
    yq = df["quality"].values.astype(float)
    groups = df["study_uid"].values
    uniq = np.unique(groups)
    by_g = {g: np.where(groups == g)[0] for g in uniq}

    def macro(org, mask):
        vals = []
        for k in ORG_LABEL_NAMES:
            y = truth[k][mask]
            p = org[k][mask]
            m = np.isfinite(p) & np.isfinite(y)
            if m.sum() and len(np.unique(y[m])) > 1:
                vals.append(f1_score(y[m], (p[m] >= 0.5).astype(int),
                                     zero_division=0))
        return float(np.mean(vals)) if vals else float("nan")

    def qba(qc, mask):
        m = mask & np.isfinite(qc) & real & ~np.isnan(yq)
        if m.sum() == 0 or len(np.unique(yq[m])) < 2:
            return float("nan")
        return balanced_accuracy_score(yq[m], qc[m].astype(int))

    d_macro, d_ba = [], []
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([by_g[g] for g in pick])
        mask = np.zeros(len(df), bool)
        mask[ii] = True
        dm = macro(pa, mask) - macro(pb, mask)
        if np.isfinite(dm):
            d_macro.append(dm)
        db = qba(qa, mask) - qba(qb, mask)
        if np.isfinite(db):
            d_ba.append(db)
    out = {}
    for tag, d in (("macro", d_macro), ("qc_ba", d_ba)):
        d = np.asarray(d)
        out[tag] = dict(mean=float(d.mean()),
                        lo=float(np.percentile(d, 2.5)),
                        hi=float(np.percentile(d, 97.5)),
                        p_gt0=float((d > 0).mean()))
    return out


tr_v_g = None


def main() -> int:
    global tr_v_g
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="cnn,fused,geo")
    ap.add_argument("--map", action="append", default=None,
                    help="полная карта источников (можно несколько раз), напр. "
                         "'femur_roi=fused_bag,spine_axis=geo_bag'")
    ap.add_argument("--protocol", default="seeds", choices=["fold_id", "seeds"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--scale", default="cnn", choices=["cnn", "clean"])
    ap.add_argument("--roi-mode", default=None,
                    choices=[None, "f1", "prior", "blend"])
    ap.add_argument("--base", default="first",
                    help="имя конфигурации-базы для бутстрэпа ('first' = первая)")
    ap.add_argument("--bootstrap", action="store_true")
    args = ap.parse_args()
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    df = load_features(load_manifest(C.MANIFEST_CSV))
    oof_q = np.load(C.OOF_QUALITY_NPY)
    oof_v = np.load(C.OOF_VIOLATION_NPY)
    tr_v = np.load(os.path.join(C.ARTIFACTS_DIR, "train_violation.npy"))
    tr_v_g = tr_v
    real = real_mask(df)

    log("реальных снимков: %d   режим порога: %s   протокол: %s   масштаб: %s"
        % (int(real.sum()), C.THRESHOLD_MODE, args.protocol, args.scale))

    base_src = dict(S.CRITERION_SOURCES)

    # Собрать список конфигураций: (имя, {критерий: источник}).
    configs = []
    if args.map:
        for spec in args.map:
            m = dict(base_src)
            for part in spec.split(","):
                if not part.strip():
                    continue
                k, v = part.split("=")
                m[k.strip()] = v.strip()
            configs.append((spec, m))
    else:
        for src in [s.strip() for s in args.sources.split(",") if s.strip()]:
            m = dict(base_src)
            m["femur_roi"] = src
            configs.append(("roi=" + src, m))

    scores, seeds_stats = {}, {}
    for name, src_map in configs:
        S.CRITERION_SOURCES.clear()
        S.CRITERION_SOURCES.update(src_map)
        _sq, stack_v, _r = S.build_scores(df, oof_q, oof_v)
        scores[name] = stack_v
        macro, sd, per, q, per_seed = evaluate_protocol(
            df, stack_v, tr_v, real, args.protocol, args.scale, args.seeds,
            args.roi_mode)
        seeds_stats[name] = per_seed
        log("")
        log("--- %s ---" % name)
        log("  macro=%.3f±%.3f | %s" % (
            macro, sd, " ".join("%s=%.3f" % (k, v) for k, v in per.items())))
        log("  quality_class: BA=%.3f macroF1=%.3f AUC=%.3f"
            % (q["ba"], q["mf1"], q["auc"]))
    S.CRITERION_SOURCES.clear()
    S.CRITERION_SOURCES.update(base_src)

    if args.bootstrap:
        log("")
        log("=== парный кластер-бутстрэп (fold_id, B=2000, scale=%s) ==="
            % args.scale)
        ref = args.base if args.base != "first" else configs[0][0]
        for name, _ in configs:
            if name == ref:
                continue
            r = paired_bootstrap(df, scores[name], scores[ref], real,
                                 args.scale, args.roi_mode)
            log("  %s - %s: macro Δ=%.3f [%.3f, %.3f] P(>0)=%.2f | "
                "qc_BA Δ=%.3f [%.3f, %.3f] P(>0)=%.2f"
                % (name, ref, r["macro"]["mean"], r["macro"]["lo"],
                   r["macro"]["hi"], r["macro"]["p_gt0"],
                   r["qc_ba"]["mean"], r["qc_ba"]["lo"], r["qc_ba"]["hi"],
                   r["qc_ba"]["p_gt0"]))

    if len(configs) > 1 and len(seeds_stats[configs[0][0]]["macro"]) > 1:
        ref = configs[0][0]
        log("")
        log("=== парно по %d сидам против '%s' (знак / Wilcoxon) ==="
            % (len(seeds_stats[ref]["macro"]), ref))
        try:
            from scipy.stats import wilcoxon
        except Exception:
            wilcoxon = None
        for name, _ in configs:
            if name == ref:
                continue
            line = "  %s:" % name
            for tag, label in (("macro", "macro"), ("ba", "qc_BA"),
                               ("mf1", "qc_mF1"), ("auc", "qc_AUC")):
                a = np.asarray(seeds_stats[name][tag])
                b = np.asarray(seeds_stats[ref][tag])
                d = a - b
                win = int((d > 0).sum())
                s = "%.3f" % float(np.mean(d))
                if wilcoxon is not None and np.any(d != 0):
                    try:
                        p = wilcoxon(a, b).pvalue
                        s += " p=%.3f" % p
                    except Exception:
                        pass
                line += " %s Δ=%s (%d/%d)" % (label, s, win, len(d))
            log(line)

    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(_LINES) + "\n")
    print("\n[exp_roi_source_real] сводка:", OUT_TXT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
