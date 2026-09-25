# -*- coding: utf-8 -*-
"""Финальное честное сравнение стратегий порога + кластер-бутстрэп по исследованиям.

Стратегии задаются картой {метка организатора: режим}. Порог метки подбирается
на train-части фолда по скору метки (org_label_score), применяется к val.
"""
from __future__ import annotations
import os, sys
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
sys.path.insert(0, os.path.abspath("."))
from src import config as C
from src.data import load_manifest, real_mask
from src.metrics import (pick_threshold, org_label_score, ORG_LABEL_NAMES,
                         org_label_truth)

df = load_manifest(C.MANIFEST_CSV)
oof_v = np.load(C.STACK_OOF_VIOLATION_NPY)
fold_id = np.load(os.path.join(C.ARTIFACTS_DIR, "fold_id.npy"))
tr_v = np.load(os.path.join(C.ARTIFACTS_DIR, "train_violation.npy"))
real = real_mask(df); folds = sorted(set(fold_id[fold_id >= 0].tolist()))
groups = df["study_uid"].values; regions = df["region"].values
yq = df["quality"].values.astype(float)

STRAT = {
    "f1":       {L: "f1" for L in ORG_LABEL_NAMES},
    "prior":    {L: "prior" for L in ORG_LABEL_NAMES},
    "blend":    {L: "blend" for L in ORG_LABEL_NAMES},
    "perlabel": {"укладка": "prior", "ось": "prior", "предметы": "prior", "ROI": "f1"},
}

def run(map_mode):
    pooled = {n: np.full(len(df), np.nan) for n in ORG_LABEL_NAMES}
    qclass = np.full(len(df), np.nan); qprob = np.full(len(df), np.nan)
    for k in folds:
        va = np.where((fold_id == k) & real)[0]; tr = (fold_id != k) & real
        thr = {}
        for L in ORG_LABEL_NAMES:
            s, app = org_label_score(df, tr_v, L); y = org_label_truth(df, L)
            m = tr & app & np.isfinite(s)
            mode = map_mode[L]
            if m.sum() >= 2 and len(np.unique(y[m])) > 1:
                t, _ = pick_threshold(y[m], s[m], mode=mode)
            else:
                t = 0.5
            for c in C.ORG_LABELS[L]:
                thr[c] = float(t)
        cp = np.full((len(va), len(C.VIOLATIONS)), np.nan)
        for i, cn in enumerate(C.VIOLATIONS):
            cp[:, i] = (oof_v[va, i] >= thr[cn]).astype(float)
        for n in ORG_LABEL_NAMES:
            mem = [cp[:, C.VIOLATION_IDX[c]] for c in C.ORG_LABELS[n]]
            pooled[n][va] = np.nanmax(np.vstack(mem), axis=0)
        for ii, i in enumerate(va):
            crit = C.REGION_CRITERIA[regions[i]]
            probs = [oof_v[i, C.VIOLATION_IDX[c]] for c in crit]
            fired = [p >= thr[c] for p, c in zip(probs, crit)]
            fin = [p for p in probs if np.isfinite(p)]
            qprob[i] = max(fin) if fin else np.nan
            qclass[i] = float(any(fired))
    return pooled, qclass, qprob

def macro_idx(idx, pooled):
    vals, per = [], {}
    for n in ORG_LABEL_NAMES:
        t = org_label_truth(df, n)[idx]; p = pooled[n][idx]
        m = np.isfinite(p)
        if len(np.unique(t[m])) < 2:
            per[n] = float("nan"); continue
        f = float(f1_score(t[m], (p[m] >= 0.5).astype(int), zero_division=0))
        per[n] = f; vals.append(f)
    return (float(np.mean(vals)) if vals else float("nan")), per

P = {name: run(mp) for name, mp in STRAT.items()}
uniq = np.unique(groups)
byg = {g: np.where(groups == g)[0] for g in uniq}

lines = ["=== честные метрики (точечно) ==="]
for name in STRAT:
    pooled, qclass, qprob = P[name]
    idx = np.where(real)[0]
    macro, per = macro_idx(idx, pooled)
    vq = np.isfinite(qclass) & real & ~np.isnan(yq)
    ba = balanced_accuracy_score(yq[vq], qclass[vq].astype(int))
    qf1 = f1_score(yq[vq], qclass[vq].astype(int), average="macro", zero_division=0)
    qauc = roc_auc_score(yq[vq], qprob[vq])
    lines.append("  %-9s macro=%.3f %s | QC BA=%.3f F1=%.3f AUC=%.3f" % (
        name, macro, {k: round(v,3) for k,v in per.items()}, ba, qf1, qauc))

# бутстрэп
rng = np.random.default_rng(0); B = 1000
boot = {name: {"macro": [], "ba": [], "qf1": [], "ROI": [], "укладка": [], "ось": [], "предметы": []} for name in STRAT}
for _ in range(B):
    pick = uniq[rng.choice(len(uniq), size=len(uniq), replace=True)]
    idx = np.concatenate([byg[g] for g in pick])
    for name in STRAT:
        pooled, qclass, qprob = P[name]
        macro, per = macro_idx(idx, pooled)
        boot[name]["macro"].append(macro)
        for k in ("ROI", "укладка", "ось", "предметы"):
            boot[name][k].append(per.get(k, np.nan))
        vq = idx[np.isfinite(qclass[idx]) & ~np.isnan(yq[idx])]
        if len(vq) and len(np.unique(yq[vq])) > 1:
            boot[name]["ba"].append(balanced_accuracy_score(yq[vq], qclass[vq].astype(int)))
            boot[name]["qf1"].append(f1_score(yq[vq], qclass[vq].astype(int), average="macro", zero_division=0))

lines.append("")
lines.append("=== бутстрэп (B=%d), mean [95%%CI] ===" % B)
for name in STRAT:
    b = boot[name]
    def s(k):
        a = np.array(b[k], dtype=float)
        return "%.3f [%.3f,%.3f]" % (np.nanmean(a), np.nanpercentile(a,2.5), np.nanpercentile(a,97.5))
    lines.append("  %-9s macro=%s QC_BA=%s QC_F1=%s" % (name, s("macro"), s("ba"), s("qf1")))
lines.append("")
lines.append("=== попарно: perlabel - prior / perlabel - f1 ===")
for base in ("prior", "f1"):
    for k in ("macro", "ba", "qf1"):
        d = np.array(boot["perlabel"][k]) - np.array(boot[base][k])
        lines.append("  perlabel-%s %s: mean=%+.3f 95%%CI=[%+.3f,%+.3f] P(>0)=%.2f" % (
            base, k, np.nanmean(d), np.nanpercentile(d,2.5), np.nanpercentile(d,97.5), np.nanmean(d>0)))
for k in ("укладка", "ось", "предметы", "ROI"):
    a = np.array(boot["perlabel"][k]); b = np.array(boot["prior"][k])
    lines.append("  %-9s perlabel=%.3f prior=%.3f" % (k, np.nanmean(a), np.nanmean(b)))

txt = "\n".join(lines)
open("results/threshold_final.txt","w",encoding="utf-8").write(txt)
print(txt.encode("ascii","replace").decode())