# -*- coding: utf-8 -*-
"""Диагностика утечки в построении глобального OOF для geo/fused источников.

Сравниваем ROC-AUC одного и того же признакового набора, посчитанный двумя способами:
  * global  — фиксированный OOF по всему набору (как в exp_fusion.build_scores);
  * honest  — переобучение внутри внешнего train-фолда, скор на внешнем val,
              затем объединение val-скоров и AUC (без порога).

Если honest << global, значит глобальный OOF завышает качество (утечка стеккинга).

Запуск:
    python scripts/diag_auc_leak.py --cnn out/cnn_resnet18,out/cnn_resnet34 --seeds 3
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dxa_qc"))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import exp_fusion as X  # noqa: E402
from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402

NAN = float("nan")


def best_f1(y, p):
    grid = np.linspace(0.02, 0.98, 193)
    return max(f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid)


def prior_f1(y, p):
    if len(y) == 0:
        return NAN
    k = max(1, int(round(float(np.mean(y)) * len(p))))
    thr = float(np.sort(p)[::-1][min(k, len(p)) - 1])
    return f1_score(y, (p >= thr).astype(int), zero_division=0)


def honest_pooled(data, real, cnn_oof, src, sets, j, name, seed):
    """val-скоры (bootstrap-bagging inner-моделей) для одного критерия/источника."""
    key = "spine" if name.startswith("spine") else "femur"
    rmask = X.region_mask(data, key)
    y = data["viol_" + name].values.astype(float)
    su = data["study_uid"].values
    add_cnn = src.startswith("fused")
    cols = [c for c in sets[name] if c in data.columns]
    Xg = data[cols].fillna(0).values
    if add_cnn:
        Xg = np.column_stack([cnn_oof[:, j], Xg])
    idx = np.where(real)[0]
    g = su[idx]
    strat = data["region"].values[idx]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    out = np.full(len(data), NAN)
    for tr_i, va_i in splitter.split(idx, strat, groups=g):
        tr, va = idx[tr_i], idx[va_i]
        known_tr = tr[rmask[tr] & ~np.isnan(y[tr])]
        va_c = va[rmask[va]]
        if len(known_tr) < 5 or len(np.unique(y[known_tr].astype(int))) < 2 or len(va_c) == 0:
            continue
        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
        acc = np.zeros(len(va_c)); cnt = 0
        for itr_i, _ in inner.split(known_tr, y[known_tr].astype(int), groups=su[known_tr]):
            itr = known_tr[itr_i]
            if len(np.unique(y[itr].astype(int))) < 2:
                continue
            m = X.clf().fit(Xg[itr], y[itr].astype(int))
            acc += m.predict_proba(Xg[va_c])[:, 1]; cnt += 1
        if cnt:
            out[va_c] = acc / cnt
    return out


def honest_pooled_macro(data, real, cnn_oof, seed):
    """Честный pooled macro-F1: refit-в-train (bagged), отбор источника по train-AUC,
    порог РАНГОВЫЙ (k = доля позитивов в train * число val в регионе) — масштабно-инвариантно."""
    idx = np.where(real)[0]
    g = data["study_uid"].values[idx]
    strat = data["region"].values[idx]
    su = data["study_uid"].values
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fired = np.zeros((len(data), len(C.VIOLATIONS)), dtype=int)
    for tr_i, va_i in splitter.split(idx, strat, groups=g):
        tr, va = idx[tr_i], idx[va_i]
        for j, name in enumerate(C.VIOLATIONS):
            key = "spine" if name.startswith("spine") else "femur"
            rmask = X.region_mask(data, key)
            y = data["viol_" + name].values.astype(float)
            known_tr = tr[rmask[tr] & ~np.isnan(y[tr])]
            if len(known_tr) < 5 or len(np.unique(y[known_tr].astype(int))) < 2:
                continue
            va_c = va[rmask[va]]
            if len(va_c) == 0:
                continue
            cand_tr, cand_va = {}, {}
            cand_tr["cnn"] = cnn_oof[:, j]; cand_va["cnn"] = cnn_oof[:, j]
            for src, sets, add in (("geo", X.GEO_SETS, False), ("geobig", X.GEO_BIG, False),
                                   ("fused", X.GEO_SETS, True), ("fusedbig", X.GEO_BIG, True)):
                cols = [c for c in sets[name] if c in data.columns]
                Xg = data[cols].fillna(0).values
                if add:
                    Xg = np.column_stack([cnn_oof[:, j], Xg])
                inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
                oof = np.full(len(data), NAN); acc = np.zeros(len(va_c)); cnt = 0
                for itr_i, iva_i in inner.split(known_tr, y[known_tr].astype(int), groups=su[known_tr]):
                    itr, iva = known_tr[itr_i], known_tr[iva_i]
                    if len(np.unique(y[itr].astype(int))) < 2:
                        continue
                    m = X.clf().fit(Xg[itr], y[itr].astype(int))
                    oof[iva] = m.predict_proba(Xg[iva])[:, 1]
                    acc += m.predict_proba(Xg[va_c])[:, 1]; cnt += 1
                if cnt == 0:
                    continue
                p_tr = np.full(len(data), NAN); p_tr[known_tr] = oof[known_tr]
                p_va = np.full(len(data), NAN); p_va[va_c] = acc / cnt
                cand_tr[src], cand_va[src] = p_tr, p_va
            best_src, best_auc = "cnn", -1.0
            for s, p in cand_tr.items():
                pp = p[known_tr]; ok = ~np.isnan(pp)
                if ok.sum() < 5 or len(np.unique(y[known_tr][ok].astype(int))) < 2:
                    continue
                a = roc_auc_score(y[known_tr][ok].astype(int), pp[ok])
                if a > best_auc:
                    best_auc, best_src = a, s
            prev = float(np.mean(y[known_tr]))
            pp_va = cand_va[best_src]
            va_ok = va[~np.isnan(pp_va[va])]
            k = max(1, int(round(prev * len(va_ok))))
            va_fire = va_ok[np.argsort(pp_va[va_ok])[::-1][:k]]
            fired[va_fire, j] = 1
    per = {}
    for lab, crits in X.ORG_LABELS.items():
        gt = np.zeros(len(data), dtype=int); pr = np.zeros(len(data), dtype=int)
        for name in crits:
            gt |= np.nan_to_num(data["viol_" + name].values.astype(float), nan=0.0).astype(int)
            pr |= fired[:, C.VIOLATION_IDX[name]]
        per[lab] = f1_score(gt[real], pr[real], zero_division=0)
    return float(np.mean(list(per.values()))), per


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn", default="out/cnn_resnet18,out/cnn_resnet34")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    dirs = [d.strip() for d in args.cnn.split(",") if d.strip()]
    oofs = [np.load(os.path.join(d, "oof_violation.npy")) for d in dirs]
    cnn_oof = np.nanmean(np.stack(oofs), axis=0)
    S = X.build_scores(data, real, cnn_oof)

    pairs = (("geo", X.GEO_SETS), ("geobig", X.GEO_BIG),
             ("fused", X.GEO_SETS), ("fusedbig", X.GEO_BIG))
    print(f"{'критерий':20} {'источник':9} | {'AUC glob':>8} {'AUC hon':>8} | "
          f"{'bestF1 glob':>11} {'bestF1 hon':>11} | {'priorF1 glob':>12} {'priorF1 hon':>12}")
    for j, name in enumerate(C.VIOLATIONS):
        key = "spine" if name.startswith("spine") else "femur"
        rmask = X.region_mask(data, key)
        y = data["viol_" + name].values.astype(float)
        for src, sets in pairs:
            pg = S[src][:, j]
            okg = real & ~np.isnan(pg) & ~np.isnan(y)
            yg = y[okg].astype(int); pgv = pg[okg]
            ag = roc_auc_score(yg, pgv) if len(np.unique(yg)) > 1 else NAN
            bfg = best_f1(yg, pgv)
            pfg = prior_f1(yg, pgv)

            aucs, bfs, pfs = [], [], []
            for seed in range(args.seeds):
                ph = honest_pooled(data, real, cnn_oof, src, sets, j, name, seed)
                okh = real & ~np.isnan(ph) & ~np.isnan(y)
                yh = y[okh].astype(int); phv = ph[okh]
                if len(np.unique(yh)) < 2:
                    continue
                aucs.append(roc_auc_score(yh, phv))
                bfs.append(best_f1(yh, phv))
                pfs.append(prior_f1(yh, phv))
            ah = float(np.mean(aucs)) if aucs else NAN
            bh = float(np.mean(bfs)) if bfs else NAN
            ph_ = float(np.mean(pfs)) if pfs else NAN
            print(f"  {name:18} {src:9} | {ag:8.3f} {ah:8.3f} | "
                  f"{bfg:11.3f} {bh:11.3f} | {pfg:12.3f} {ph_:12.3f}")

    print("\n=== Честный pooled macro-F1 (refit-в-train, ранговый порог) ===")
    vals, pers = [], {k: [] for k in X.ORG_LABELS}
    for seed in range(args.seeds):
        m, per = honest_pooled_macro(data, real, cnn_oof, seed)
        vals.append(m)
        for k in per:
            pers[k].append(per[k])
    per_s = " ".join(f"{k}={np.mean(v):.3f}" for k, v in pers.items())
    print(f"  macro-F1={np.mean(vals):.3f}±{np.std(vals):.3f}   {per_s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
