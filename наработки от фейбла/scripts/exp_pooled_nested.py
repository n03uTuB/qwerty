# -*- coding: utf-8 -*-
"""Полностью честная pooled-оценка (метрика организатора).

Отличия от exp_pooled.py — убраны ОБА подглядывания:
  * geo/fused/geobig/fusedbig ПЕРЕОБУЧАЮТСЯ внутри каждого train-фолда
    (нет кросс-загрязнения фиксированного OOF по внешним фолдам);
  * источник по критерию выбирается по train-OOF AUC, а не по AUC на всём наборе;
  * порог — по train-OOF (prior|f1|blend).

Оговорка: cnn_oof — фиксированный OOF CNN (свой фолд-сплит, seed 42), поэтому для
источника 'cnn' кросс-загрязнение остаётся (как и в harness команды). Это нижняя
граница честности без переобучения CNN на каждый внешний фолд.

Метрика — pooled macro-F1 по 4 меткам; усреднение по seeds разбиений;
парные разности — бутстрэп по seeds.

Запуск:
    python scripts/exp_pooled_nested.py --cnn out/cnn_resnet18,out/cnn_resnet34 --seeds 10
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


def fold_fired(data, real, cnn_oof, tr, va, thr_mode):
    """Один фолд: переобучение источников + nested-отбор + порог — всё по train."""
    n = len(data)
    fired = np.zeros((n, len(C.VIOLATIONS)), dtype=int)
    su = data["study_uid"].values
    for j, name in enumerate(C.VIOLATIONS):
        key = "spine" if name.startswith("spine") else "femur"
        rmask = X.region_mask(data, key)
        y = data["viol_" + name].values.astype(float)
        known_tr = tr[rmask[tr] & ~np.isnan(y[tr])]
        if len(known_tr) < 3 or len(np.unique(y[known_tr].astype(int))) < 2:
            continue

        cand_tr, cand_va = {}, {}
        cand_tr["cnn"] = cnn_oof[:, j]
        cand_va["cnn"] = cnn_oof[:, j]
        for src, sets in (("geo", X.GEO_SETS), ("geobig", X.GEO_BIG)):
            cols = [c for c in sets[name] if c in data.columns]
            Xg = data[cols].fillna(0).values
            oof = X.cv_score(Xg[known_tr], y[known_tr].astype(int), su[known_tr])
            p_tr = np.full(n, NAN); p_tr[known_tr] = oof
            m = X.clf().fit(Xg[known_tr], y[known_tr].astype(int))
            va_c = va[rmask[va]]
            p_va = np.full(n, NAN)
            p_va[va_c] = m.predict_proba(Xg[va_c])[:, 1]
            cand_tr[src], cand_va[src] = p_tr, p_va
        for src, sets in (("fused", X.GEO_SETS), ("fusedbig", X.GEO_BIG)):
            cols = [c for c in sets[name] if c in data.columns]
            Xg = np.column_stack([cnn_oof[:, j], data[cols].fillna(0).values])
            oof = X.cv_score(Xg[known_tr], y[known_tr].astype(int), su[known_tr])
            p_tr = np.full(n, NAN); p_tr[known_tr] = oof
            m = X.clf().fit(Xg[known_tr], y[known_tr].astype(int))
            va_c = va[rmask[va]]
            p_va = np.full(n, NAN)
            p_va[va_c] = m.predict_proba(Xg[va_c])[:, 1]
            cand_tr[src], cand_va[src] = p_tr, p_va

        # nested-отбор источника по train-OOF AUC (без валидации)
        best_src, best_auc = "cnn", -1.0
        for s, p in cand_tr.items():
            pp = p[known_tr]
            ok = ~np.isnan(pp)
            if ok.sum() < 3 or len(np.unique(y[known_tr][ok].astype(int))) < 2:
                continue
            a = roc_auc_score(y[known_tr][ok].astype(int), pp[ok])
            if a > best_auc:
                best_auc, best_src = a, s

        pp_tr = cand_tr[best_src][known_tr]
        ok = ~np.isnan(pp_tr)
        ytr = y[known_tr][ok].astype(int)
        ptr = pp_tr[ok]
        if len(ytr) < 2 or len(np.unique(ytr)) < 2:
            continue
        if thr_mode in ("kprior", "kf1", "kblend"):
            k = X.pick_k(ytr, ptr, thr_mode)
            pp_va = cand_va[best_src]
            va_ok = va[~np.isnan(pp_va[va])]
            va_ok = va_ok[np.argsort(pp_va[va_ok])[::-1][:k]]
            fired[va_ok, j] = 1
        elif thr_mode == "prank":
            pp_va = cand_va[best_src]
            va_ok = va[~np.isnan(pp_va[va])]
            prev = float(np.mean(ytr))
            k = max(1, int(round(prev * len(va_ok))))
            va_ok = va_ok[np.argsort(pp_va[va_ok])[::-1][:k]]
            fired[va_ok, j] = 1
        else:
            thr = X.pick_threshold(ytr, ptr, thr_mode)
            pp_va = cand_va[best_src]
            va_ok = va[~np.isnan(pp_va[va])]
            fired[va_ok[pp_va[va_ok] >= thr], j] = 1
    return fired


def fold_fired_bagged(data, real, cnn_oof, tr, va, thr_mode):
    """Строгий nested: val-скоры geo/fused — бэггинг inner-моделей (то же
    распределение, что и inner-OOF для порога). CNN остаётся глобальным OOF."""
    n = len(data)
    fired = np.zeros((n, len(C.VIOLATIONS)), dtype=int)
    su = data["study_uid"].values
    for j, name in enumerate(C.VIOLATIONS):
        key = "spine" if name.startswith("spine") else "femur"
        rmask = X.region_mask(data, key)
        y = data["viol_" + name].values.astype(float)
        known_tr = tr[rmask[tr] & ~np.isnan(y[tr])]
        if len(known_tr) < 5 or len(np.unique(y[known_tr].astype(int))) < 2:
            continue
        va_c = va[rmask[va]]
        cand_tr, cand_va = {}, {}
        cand_tr["cnn"] = cnn_oof[:, j]
        cand_va["cnn"] = cnn_oof[:, j]
        for src, sets, add_cnn in (("geo", X.GEO_SETS, False), ("geobig", X.GEO_BIG, False),
                                   ("fused", X.GEO_SETS, True), ("fusedbig", X.GEO_BIG, True)):
            cols = [c for c in sets[name] if c in data.columns]
            Xg = data[cols].fillna(0).values
            if add_cnn:
                Xg = np.column_stack([cnn_oof[:, j], Xg])
            inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
            oof = np.full(n, NAN)
            acc = np.zeros(len(va_c))
            cnt = 0
            for itr_i, iva_i in inner.split(known_tr, y[known_tr].astype(int),
                                            groups=su[known_tr]):
                itr, iva = known_tr[itr_i], known_tr[iva_i]
                if len(np.unique(y[itr].astype(int))) < 2:
                    continue
                m = X.clf().fit(Xg[itr], y[itr].astype(int))
                oof[iva] = m.predict_proba(Xg[iva])[:, 1]
                acc += m.predict_proba(Xg[va_c])[:, 1]
                cnt += 1
            if cnt == 0:
                continue
            p_tr = np.full(n, NAN); p_tr[known_tr] = oof[known_tr]
            p_va = np.full(n, NAN); p_va[va_c] = acc / cnt
            cand_tr[src], cand_va[src] = p_tr, p_va

        best_src, best_auc = "cnn", -1.0
        for s, p in cand_tr.items():
            pp = p[known_tr]; ok = ~np.isnan(pp)
            if ok.sum() < 5 or len(np.unique(y[known_tr][ok].astype(int))) < 2:
                continue
            a = roc_auc_score(y[known_tr][ok].astype(int), pp[ok])
            if a > best_auc:
                best_auc, best_src = a, s
        pp_tr = cand_tr[best_src][known_tr]; ok = ~np.isnan(pp_tr)
        ytr = y[known_tr][ok].astype(int); ptr = pp_tr[ok]
        if len(ytr) < 2 or len(np.unique(ytr)) < 2:
            continue
        pp_va = cand_va[best_src]
        va_ok = va[~np.isnan(pp_va[va])]
        if thr_mode in ("kprior", "kf1", "kblend"):
            k = X.pick_k(ytr, ptr, thr_mode)
            va_ok = va_ok[np.argsort(pp_va[va_ok])[::-1][:k]]
            fired[va_ok, j] = 1
        elif thr_mode == "prank":
            # калибровочно-устойчивый prior: верхние k по РАНГУ val-скоров,
            # k = доля позитивов в train * число val-снимков в регионе
            prev = float(np.mean(ytr))
            k = max(1, int(round(prev * len(va_ok))))
            va_ok = va_ok[np.argsort(pp_va[va_ok])[::-1][:k]]
            fired[va_ok, j] = 1
        else:
            thr = X.pick_threshold(ytr, ptr, thr_mode)
            fired[va_ok[pp_va[va_ok] >= thr], j] = 1
    return fired


def pooled_seed_bagged(data, real, cnn_oof, seed, thr_mode):
    idx = np.where(real)[0]
    g = data["study_uid"].values[idx]
    strat = data["region"].values[idx]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fired = np.zeros((len(data), len(C.VIOLATIONS)), dtype=int)
    for tr_i, va_i in splitter.split(idx, strat, groups=g):
        tr, va = idx[tr_i], idx[va_i]
        f = fold_fired_bagged(data, real, cnn_oof, tr, va, thr_mode)
        fired[va] = f[va]
    return _pooled_macro(data, real, fired)


def pooled_seed(data, real, cnn_oof, seed, thr_mode):
    """refit внутри train: geo/fused переобучаются в каждом фолде."""
    idx = np.where(real)[0]
    g = data["study_uid"].values[idx]
    strat = data["region"].values[idx]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fired = np.zeros((len(data), len(C.VIOLATIONS)), dtype=int)
    for tr_i, va_i in splitter.split(idx, strat, groups=g):
        tr, va = idx[tr_i], idx[va_i]
        f = fold_fired(data, real, cnn_oof, tr, va, thr_mode)
        fired[va] = f[va]
    return _pooled_macro(data, real, fired)


def fold_map_global(data, real, S, tr, va, thr_mode):
    """Глобальные OOF-скоры, но карта источников выбирается ТОЛЬКО по train-AUC."""
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
    return X.fired_from(data, real, S, source_map, tr, va, thr_mode)


def pooled_seed_global(data, real, S, seed, thr_mode):
    idx = np.where(real)[0]
    g = data["study_uid"].values[idx]
    strat = data["region"].values[idx]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fired = np.zeros((len(data), len(C.VIOLATIONS)), dtype=int)
    for tr_i, va_i in splitter.split(idx, strat, groups=g):
        tr, va = idx[tr_i], idx[va_i]
        f = fold_map_global(data, real, S, tr, va, thr_mode)
        fired[va] = f[va]
    return _pooled_macro(data, real, fired)


def pooled_seed_fixed(data, real, S, source_map, seed, thr_mode):
    """Глобальные OOF-скоры + фиксированная карта источников (как exp_pooled)."""
    idx = np.where(real)[0]
    g = data["study_uid"].values[idx]
    strat = data["region"].values[idx]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fired = np.zeros((len(data), len(C.VIOLATIONS)), dtype=int)
    for tr_i, va_i in splitter.split(idx, strat, groups=g):
        tr, va = idx[tr_i], idx[va_i]
        f = X.fired_from(data, real, S, source_map, tr, va, thr_mode)
        fired[va] = f[va]
    return _pooled_macro(data, real, fired)


def _pooled_macro(data, real, fired):
    per = {}
    for lab, crits in X.ORG_LABELS.items():
        gt = np.zeros(len(data), dtype=int)
        pr = np.zeros(len(data), dtype=int)
        for name in crits:
            gt |= np.nan_to_num(data["viol_" + name].values.astype(float),
                                nan=0.0).astype(int)
            pr |= fired[:, C.VIOLATION_IDX[name]]
        per[lab] = f1_score(gt[real], pr[real], zero_division=0)
    return float(np.mean(list(per.values()))), per


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn", default="out/cnn_resnet18,out/cnn_resnet34")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--modes", default="ABCD")
    args = ap.parse_args()

    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    dirs = [d.strip() for d in args.cnn.split(",") if d.strip()]
    oofs = [np.load(os.path.join(d, "oof_violation.npy")) for d in dirs]
    cnn_oof = np.nanmean(np.stack(oofs), axis=0)
    print(f"[pooled-nested] строк={len(data)} реальных={int(real.sum())} CNN={dirs}")

    S = X.build_scores(data, real, cnn_oof)
    team_map = {"spine_positioning": "cnn", "spine_axis": "fused",
                "spine_artifacts": "cnn", "femur_positioning": "cnn",
                "femur_roi": "cnn"}

    def run(fn):
        vals, pers = [], {k: [] for k in X.ORG_LABELS}
        for seed in range(args.seeds):
            m, per = fn(seed)
            vals.append(m)
            for k in per:
                pers[k].append(per[k])
        return np.array(vals), pers

    results = {}

    print("\n=== A. refit внутри train + nested-отбор (строгая нижняя граница) ===")
    if "A" in args.modes:
        for tmode in ("prior", "f1", "blend", "prank"):
            v, pers = run(lambda s, t=tmode: pooled_seed(data, real, cnn_oof, s, t))
            results[("refit", tmode)] = v
            per_s = " ".join(f"{k}={np.mean(x):.3f}" for k, x in pers.items())
            print(f"  {tmode:12} {v.mean():10.3f} {v.std():8.3f}   {per_s}")

    print("\n=== B. глобальные OOF + nested-отбор карты (честная верхняя граница) ===")
    if "B" in args.modes:
        for tmode in ("prior", "f1", "blend", "prank"):
            v, pers = run(lambda s, t=tmode: pooled_seed_global(data, real, S, s, t))
            results[("global", tmode)] = v
            per_s = " ".join(f"{k}={np.mean(x):.3f}" for k, x in pers.items())
            print(f"  {tmode:12} {v.mean():10.3f} {v.std():8.3f}   {per_s}")

    print("\n=== C. глобальные OOF + фиксированная карта команды (для сверки с exp_pooled) ===")
    if "C" in args.modes:
        for tmode in ("prior", "blend", "prank"):
            v, pers = run(lambda s, t=tmode: pooled_seed_fixed(data, real, S, team_map, s, t))
            results[("team", tmode)] = v
            per_s = " ".join(f"{k}={np.mean(x):.3f}" for k, x in pers.items())
            print(f"  {tmode:12} {v.mean():10.3f} {v.std():8.3f}   {per_s}")

    print("\n=== D. строгий nested: бэггинг inner-моделей (распределения согласованы) ===")
    if "D" in args.modes:
        for tmode in ("prior", "f1", "blend", "prank"):
            v, pers = run(lambda s, t=tmode: pooled_seed_bagged(data, real, cnn_oof, s, t))
            results[("bagged", tmode)] = v
            per_s = " ".join(f"{k}={np.mean(x):.3f}" for k, x in pers.items())
            print(f"  {tmode:12} {v.mean():10.3f} {v.std():8.3f}   {per_s}")

    print("\nпарные разности (бутстрэп по seeds):")
    rng = np.random.default_rng(0)

    def cmp(a, b, label):
        if a not in results or b not in results:
            return
        diff = results[a] - results[b]
        boots = [np.mean(rng.choice(diff, len(diff), replace=True)) for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        sig = "ЗНАЧИМО" if (lo > 0 or hi < 0) else "шум"
        print(f"  {label:38} Δ={diff.mean():+.3f} [{lo:+.3f};{hi:+.3f}]  {sig}")

    cmp(("global", "blend"), ("team", "prior"), "B(blend) - C(team prior)")
    cmp(("global", "blend"), ("global", "prior"), "B: blend - prior")
    cmp(("global", "prank"), ("bagged", "prank"), "B(prank) - D(prank)  [калибровка]")
    cmp(("global", "blend"), ("bagged", "blend"), "B(blend) - D(blend)")
    cmp(("global", "prank"), ("global", "prior"), "B: prank - prior")
    cmp(("bagged", "prank"), ("bagged", "prior"), "D: prank - prior")
    cmp(("global", "blend"), ("global", "prank"), "B: blend - prank")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
