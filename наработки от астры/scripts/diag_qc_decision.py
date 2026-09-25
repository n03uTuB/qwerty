# -*- coding: utf-8 -*-
"""QC-решение: OR(критериев) vs порог по quality_prob на регион (честно nested)."""
from __future__ import annotations
import os, sys
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
sys.path.insert(0, os.path.abspath("."))
from src import config as C
from src.data import load_manifest, real_mask
from src.metrics import pick_threshold, prior_threshold, best_threshold

df = load_manifest(C.MANIFEST_CSV)
oof_v = np.load(C.STACK_OOF_VIOLATION_NPY)
fold_id = np.load(os.path.join(C.ARTIFACTS_DIR, "fold_id.npy"))
tr_v = np.load(os.path.join(C.ARTIFACTS_DIR, "train_violation.npy"))
real = real_mask(df); folds = sorted(set(fold_id[fold_id >= 0].tolist()))
regions = df["region"].values
yq = df["quality"].values.astype(float)

def qprob_of(probs, region):
    crit = C.REGION_CRITERIA[region]
    vals = [probs[C.VIOLATION_IDX[c]] for c in crit if np.isfinite(probs[C.VIOLATION_IDX[c]])]
    return max(vals) if vals else np.nan

# quality_prob на уровне строк (train и oof)
qp_oof = np.array([qprob_of(oof_v[i], regions[i]) for i in range(len(df))])
qp_tr  = np.array([qprob_of(tr_v[i], regions[i]) for i in range(len(df))])

def qc_metrics(pred, prob):
    v = np.isfinite(pred) & real & ~np.isnan(yq)
    return dict(ba=float(balanced_accuracy_score(yq[v], pred[v].astype(int))),
                f1=float(f1_score(yq[v], pred[v].astype(int), average="macro", zero_division=0)),
                auc=float(roc_auc_score(yq[v], prob[v])))

def eval_or(ccmode):
    """QC = OR(критериев) с порогами ccmode по train."""
    pred = np.full(len(df), np.nan)
    for k in folds:
        va = np.where((fold_id == k) & real)[0]; tr = (fold_id != k) & real
        thr = {}
        for i, cn in enumerate(C.VIOLATIONS):
            ytr = df["viol_" + cn].values.astype(float)[tr]; ptr = tr_v[tr, i]
            ok = ~np.isnan(ytr) & ~np.isnan(ptr)
            thr[cn] = pick_threshold(ytr[ok], ptr[ok], mode=ccmode)[0] if (ok.sum() and len(np.unique(ytr[ok]))>1) else 0.5
        for i in va:
            crit = C.REGION_CRITERIA[regions[i]]
            fired = [oof_v[i, C.VIOLATION_IDX[c]] >= thr[c] for c in crit]
            pred[i] = float(any(fired))
    return pred

def eval_prob(obj):
    """QC = (quality_prob >= порог на регион), порог по train под obj (f1|ba)."""
    pred = np.full(len(df), np.nan)
    for k in folds:
        va = np.where((fold_id == k) & real)[0]; tr = (fold_id != k) & real
        for reg in C.REGIONS:
            mtr = tr & (regions == reg) & ~np.isnan(yq) & np.isfinite(qp_tr)
            if mtr.sum() >= 2 and len(np.unique(yq[mtr])) > 1:
                if obj == "ba":
                    best, bt = -1, 0.5
                    for t in np.linspace(0.02, 0.98, 97):
                        b = balanced_accuracy_score(yq[mtr], (qp_tr[mtr] >= t).astype(int))
                        if b > best: best, bt = b, float(t)
                    t = bt
                else:
                    t, _ = pick_threshold(yq[mtr], qp_tr[mtr], mode="f1")
            else:
                t = 0.5
            mva = (fold_id == k) & real & (regions == reg)
            pred[mva] = (qp_oof[mva] >= t).astype(float)
    return pred

lines = []
for name, pred in [("OR(f1)", eval_or("f1")), ("OR(prior)", eval_or("prior")),
                   ("prob-f1", eval_prob("f1")), ("prob-BA", eval_prob("ba"))]:
    m = qc_metrics(pred, qp_oof)
    lines.append("  %-10s QC BA=%.3f macroF1=%.3f AUC=%.3f" % (name, m["ba"], m["f1"], m["auc"]))
txt = "=== QC-решение ===\n" + "\n".join(lines)
open("results/qc_decision.txt","w",encoding="utf-8").write(txt)
print(txt.encode("ascii","replace").decode())