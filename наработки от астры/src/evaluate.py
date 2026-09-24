# -*- coding: utf-8 -*-
"""Оценка гибридного решения: Balanced Accuracy, Macro-F1, ROC-AUC с 95% ДИ.

Считает три блока:
  1. голова качества (CNN) — ROC-AUC/AP с ДИ (кластер-бутстрэп по study_uid);
  2. ``quality_class`` = ИЛИ(сработавших критериев) — так же, как в инференсе:
     Balanced Accuracy / Macro-F1 / ROC-AUC (quality_prob = max по критериям области);
  3. метрика организатора — macro-F1 по 4 меткам, пороги выбраны ТОЛЬКО по
     train-части фолда (честно), затем применены к val-части и пулятся.

Режим порога задаётся ``--thr`` (иначе config.THRESHOLD_MODE). Результат —
artifacts/evaluate.json + artifacts/report.md.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, roc_auc_score)

from . import config as C
from .data import load_manifest, real_mask
from .metrics import (ORG_LABEL_NAMES, bootstrap_ci, bootstrap_ci_by_study,
                      org_label_truth, pick_threshold)

EVAL_JSON = os.path.join(C.ARTIFACTS_DIR, "evaluate.json")
REPORT_MD = os.path.join(C.ARTIFACTS_DIR, "report.md")


def _ci(y, p, groups, metric, thr=0.5):
    return dict(
        object=bootstrap_ci(y, p, metric, thr=thr),
        study=bootstrap_ci_by_study(y, p, groups, metric, thr=thr),
    )


def _quality_class_from_criteria(df, oof_v, tr_v, fold_id, mode, real):
    """quality_prob/quality_class как в инференсе, с честными порогами по train."""
    n = len(df)
    q_prob = np.full(n, np.nan)
    q_class = np.full(n, np.nan)
    fold_ids = sorted(set(fold_id[fold_id >= 0].tolist()))
    thr_by_fold = {}
    for k in fold_ids:
        tr = np.where((fold_id != k) & real & ~np.isnan(tr_v[:, 0]))[0]
        thr_by_fold[k] = {}
        for i, name in enumerate(C.VIOLATIONS):
            y_tr = df["viol_" + name].values.astype(float)[tr]
            p_tr = tr_v[tr, i]
            ok = ~np.isnan(y_tr) & ~np.isnan(p_tr)
            if ok.sum() and len(np.unique(y_tr[ok])) > 1:
                t, _ = pick_threshold(y_tr[ok], p_tr[ok], mode=mode)
            else:
                t = 0.5
            thr_by_fold[k][name] = t
    regions = df["region"].values
    for k in fold_ids:
        va = np.where((fold_id == k) & real)[0]
        for i in va:
            crit = C.REGION_CRITERIA[regions[i]]
            probs = [oof_v[i, C.VIOLATION_IDX[c]] for c in crit]
            fired = [p >= thr_by_fold[k][c] for p, c in zip(probs, crit)]
            finite = [p for p in probs if np.isfinite(p)]
            q_prob[i] = max(finite) if finite else np.nan
            q_class[i] = float(any(fired))
    return q_prob, q_class


def _organizer_honest(df, oof_v, tr_v, fold_id, mode, real):
    truth = {name: org_label_truth(df, name) for name in ORG_LABEL_NAMES}
    pooled_pred = {name: np.full(len(df), np.nan) for name in ORG_LABEL_NAMES}
    fold_ids = sorted(set(fold_id[fold_id >= 0].tolist()))
    for k in fold_ids:
        va = np.where((fold_id == k) & real)[0]
        tr = np.where((fold_id != k) & real & ~np.isnan(tr_v[:, 0]))[0]
        if len(va) == 0:
            continue
        crit_pred = {}
        for i, cname in enumerate(C.VIOLATIONS):
            y_tr = df["viol_" + cname].values.astype(float)[tr]
            p_tr = tr_v[tr, i]
            ok = ~np.isnan(y_tr) & ~np.isnan(p_tr)
            if ok.sum() and len(np.unique(y_tr[ok])) > 1:
                thr, _ = pick_threshold(y_tr[ok], p_tr[ok], mode=mode)
            else:
                thr = 0.5
            crit_pred[cname] = (oof_v[va, i] >= thr).astype(float)
        for name in ORG_LABEL_NAMES:
            members = [crit_pred[c] for c in C.ORG_LABELS[name]]
            pooled_pred[name][va] = np.max(np.vstack(members), axis=0)
    per, vals = {}, []
    for name in ORG_LABEL_NAMES:
        m = np.isfinite(pooled_pred[name])
        y, p = truth[name][m], pooled_pred[name][m]
        if len(y) == 0 or len(np.unique(y)) < 2:
            per[name] = float("nan")
            continue
        f = float(f1_score(y, (p >= 0.5).astype(int), zero_division=0))
        per[name] = f
        vals.append(f)
    return dict(macro=float(np.mean(vals)) if vals else float("nan"),
                per_label=per)


def main():
    ap = argparse.ArgumentParser(description="Оценка гибридного DXA-QC решения")
    ap.add_argument("--thr", default=None,
                    choices=["f1", "prior", "blend", "nested"],
                    help="режим порога (по умолчанию config.THRESHOLD_MODE)")
    ap.add_argument("--manifest", default=C.MANIFEST_CSV)
    ap.add_argument("--stacked", action="store_true",
                    help="использовать стекер-шкалу (stack_oof_violation.npy)")
    args = ap.parse_args()
    mode = args.thr or C.THRESHOLD_MODE

    df = load_manifest(args.manifest)
    oof_q = np.load(C.OOF_QUALITY_NPY)
    oof_v = np.load(C.OOF_VIOLATION_NPY)
    if args.stacked and os.path.isfile(C.STACK_OOF_VIOLATION_NPY):
        oof_v = np.load(C.STACK_OOF_VIOLATION_NPY)
    fold_id = np.load(os.path.join(C.ARTIFACTS_DIR, "fold_id.npy"))
    tr_v = np.load(os.path.join(C.ARTIFACTS_DIR, "train_violation.npy"))
    real = real_mask(df)
    groups = df["study_uid"].values
    out = {"threshold_mode": mode, "n": int(real.sum())}

    # --- 1. голова качества (CNN) ---
    y = df["quality"].values.astype(float)
    m = ~np.isnan(y) & ~np.isnan(oof_q) & real
    if m.sum() and len(np.unique(y[m])) > 1:
        out["quality_head"] = dict(
            n=int(m.sum()), pos=int((y[m] > 0.5).sum()),
            auc=float(roc_auc_score(y[m], oof_q[m])),
            ap=float(average_precision_score(y[m], oof_q[m])),
            auc_ci=_ci(y[m], oof_q[m], groups[m], "auc"),
        )

    # --- 2. quality_class = OR(критериев) ---
    q_prob, q_class = _quality_class_from_criteria(df, oof_v, tr_v, fold_id, mode, real)
    vq = np.isfinite(q_class) & real
    yq = df["quality"].values.astype(float)
    vq &= ~np.isnan(yq)
    if vq.sum() and len(np.unique(yq[vq])) > 1:
        pred = q_class[vq].astype(int)
        out["quality_class"] = dict(
            n=int(vq.sum()), pos=int((yq[vq] > 0.5).sum()),
            balanced_accuracy=float(balanced_accuracy_score(yq[vq], pred)),
            macro_f1=float(f1_score(yq[vq], pred, average="macro", zero_division=0)),
            auc=float(roc_auc_score(yq[vq], q_prob[vq])),
            auc_ci=_ci(yq[vq], q_prob[vq], groups[vq], "auc"),
            f1_ci=_ci(yq[vq], q_prob[vq], groups[vq], "f1",
                      thr=float(np.median(q_prob[vq]))),
        )

    # --- 3. метрика организатора (честно) ---
    out["organizer"] = _organizer_honest(df, oof_v, tr_v, fold_id, mode, real)

    # --- 4. по критериям ---
    per = {}
    for i, name in enumerate(C.VIOLATIONS):
        yv = df["viol_" + name].values.astype(float)
        v = ~np.isnan(yv) & ~np.isnan(oof_v[:, i]) & real
        if v.sum() == 0 or len(np.unique(yv[v])) < 2:
            per[name] = dict(n=int(v.sum()), f1=None, auc=None)
            continue
        thr, f1 = pick_threshold(yv[v], oof_v[v, i], mode=mode)
        per[name] = dict(
            n=int(v.sum()), pos=int((yv[v] > 0.5).sum()),
            f1=float(f1), auc=float(roc_auc_score(yv[v], oof_v[v, i])),
            thr=float(thr),
            auc_ci=bootstrap_ci_by_study(yv[v], oof_v[v, i], groups[v], "auc"),
        )
    out["violations"] = per

    os.makedirs(C.ARTIFACTS_DIR, exist_ok=True)
    with open(EVAL_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)

    _write_report(out)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=float))
    print("\n[evaluate] сохранено:", EVAL_JSON, "и", REPORT_MD)


def _fmt_ci(ci):
    """Форматировать ДИ: поддерживает и dict (object/study), и просто tuple."""
    if not ci:
        return "-"
    if isinstance(ci, dict):
        ci = ci.get("study", ci.get("object"))
    if not ci or len(ci) < 2:
        return "-"
    lo, hi = ci[0], ci[1]
    if lo != lo or hi != hi:      # NaN
        return "-"
    return "[%.3f, %.3f]" % (lo, hi)


def _write_report(out: dict):
    L = ["# Отчёт по качеству гибридного DXA-QC", ""]
    L.append("Режим порога: **%s**   Реальных снимков: **%d**" %
             (out.get("threshold_mode"), out.get("n", 0)))
    L.append("")
    qc = out.get("quality_class")
    if qc:
        L += ["## quality_class (= ИЛИ критериев)", "",
              "| Метрика | Значение | 95% ДИ (по исследованиям) |",
              "|---|---|---|",
              "| Balanced Accuracy | %.3f | - |" % qc["balanced_accuracy"],
              "| Macro-F1 | %.3f | %s |" % (qc["macro_f1"], _fmt_ci(qc.get("f1_ci"))),
              "| ROC-AUC | %.3f | %s |" % (qc["auc"], _fmt_ci(qc.get("auc_ci"))),
              ""]
    org = out.get("organizer")
    if org:
        L += ["## Метрика организатора (macro-F1 по 4 меткам, честно)", "",
              "| Метка | F1 |", "|---|---|"]
        for name, f in org.get("per_label", {}).items():
            L.append("| %s | %s |" % (name, "-" if f != f else "%.3f" % f))
        L.append("| **macro-F1** | **%.3f** |" % org.get("macro", float("nan")))
        L.append("")
    qh = out.get("quality_head")
    if qh:
        L += ["## Голова качества (CNN)", "",
              "| Метрика | Значение | 95% ДИ (по исследованиям) |",
              "|---|---|---|",
              "| ROC-AUC | %.3f | %s |" % (qh["auc"], _fmt_ci(qh.get("auc_ci"))),
              "| PR-AUC | %.3f | - |" % qh["ap"], ""]
    L += ["## По критериям", "",
          "| Критерий | N | Pos | F1 | ROC-AUC | 95% ДИ AUC | Порог |",
          "|---|---|---|---|---|---|---|"]
    for name, m in out.get("violations", {}).items():
        if m.get("f1") is None:
            L.append("| %s | %d | - | - | - | - | - |" % (name, m["n"]))
            continue
        L.append("| %s | %d | %d | %.3f | %.3f | %s | %.3f |" % (
            name, m["n"], m["pos"], m["f1"], m["auc"], _fmt_ci(m.get("auc_ci")),
            m["thr"]))
    L.append("")
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()