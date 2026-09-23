# -*- coding: utf-8 -*-
"""Итерация 3: (A) парная проверка femur_positioning->geo, (B) совместный порог.

(A) Перебор источников (select_sources_honest.py) намекнул, что для
    femur_positioning чистая геометрия даёт +0.013 macro-F1 к базе. Прибавка
    внутри шума, а сам выбор сделан на тех же 20 разбиениях, где измеряется, —
    поэтому здесь сравнение ПАРНОЕ по разбиениям: на каждом seed считаем обе
    конфигурации и смотрим знак разницы и долю выигрышных разбиений.

(B) Пороги в пайплайне подбираются по каждому критерию независимо, максимизируя
    его собственный F1. Но метрика организатора — macro-F1 по 4 меткам, каждая из
    которых есть OR своих критериев (укладка = OR(spine_positioning,
    femur_positioning)). Поэтому пороги можно подбирать СОВМЕСТНО, максимизируя
    именно macro-F1 на train-части фолда (покоординатный подъём по сетке).
    Сравниваем с независимым подбором на тех же разбиениях.

Запуск:
    cd dxa_qc && python ../scripts/exp_iter3.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dxa_qc"))

from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402
from src import stack as st        # noqa: E402
from src.train import pick_threshold  # noqa: E402

import eval_honest_organizer as E  # noqa: E402

N_SEEDS = 20
GRID = np.linspace(0.05, 0.95, 91)


def _splits(data, real, seed):
    idx = np.where(real)[0]
    g = data["study_uid"].values[idx]
    strat = data["region_idx"].values[idx] if "region_idx" in data.columns else None
    if strat is None or len(np.unique(strat)) < 2:
        return idx, GroupKFold(n_splits=5).split(idx, groups=g)
    return idx, StratifiedGroupKFold(n_splits=5, shuffle=True,
                                     random_state=seed).split(idx, strat, groups=g)


def _macro(data, idx, fired, per_label=False):
    fs, per = [], {}
    for lab, crits in E.ORG_LABELS.items():
        gt = np.zeros(len(data), dtype=int)
        pr = np.zeros(len(data), dtype=int)
        for name in crits:
            v = data["viol_" + name].values.astype(float)
            gt |= np.nan_to_num(v, nan=0.0).astype(int)
            pr |= fired[:, C.VIOLATION_IDX[name]]
        f = f1_score(gt[idx], pr[idx], zero_division=0)
        fs.append(f)
        per[lab] = f
    return (float(np.mean(fs)), per) if per_label else float(np.mean(fs))


def _per_criterion_thresholds(data, tr, S, sm):
    """Порог для каждого критерия независимо (как в пайплайне)."""
    thr = {}
    for j, name in enumerate(C.VIOLATIONS):
        key = "spine" if name.startswith("spine") else "femur"
        rmask = data["region"].str.contains(key).values
        y = data["viol_" + name].values.astype(float)
        p = S[sm[name]][:, j]
        trc = tr[rmask[tr] & ~np.isnan(y[tr]) & ~np.isnan(p[tr])]
        if len(trc) < 2 or len(np.unique(y[trc].astype(int))) < 2:
            thr[name] = 0.5
            continue
        thr[name], _ = pick_threshold(y[trc].astype(int), p[trc], C.THRESHOLD_MODE)
    return thr


def _fire(data, rows, S, sm, thr):
    fired = np.zeros((len(data), len(C.VIOLATIONS)), dtype=int)
    for j, name in enumerate(C.VIOLATIONS):
        p = S[sm[name]][rows, j]
        ok = ~np.isnan(p)
        fired[rows[ok][p[ok] >= thr[name]], j] = 1
    return fired


def _joint_thresholds(data, tr, S, sm, rounds=2):
    """Покоординатный подъём порогов на train, максимизируя macro-F1 организатора."""
    thr = _per_criterion_thresholds(data, tr, S, sm)
    ytr = {}
    for name in C.VIOLATIONS:
        ytr[name] = data["viol_" + name].values.astype(float)

    def score(thr):
        fired = _fire(data, tr, S, sm, thr)
        return _macro(data, tr, fired)

    best = score(thr)
    for _ in range(rounds):
        improved = False
        for name in C.VIOLATIONS:
            j = C.VIOLATION_IDX[name]
            p = S[sm[name]][:, j]
            cand = p[tr][~np.isnan(p[tr])]
            if len(cand) < 2:
                continue
            cur = thr[name]
            for t in GRID:
                if t == cur:
                    continue
                thr[name] = float(t)
                s = score(thr)
                if s > best + 1e-9:
                    best, cur, improved = s, float(t), True
                else:
                    thr[name] = cur
        if not improved:
            break
    return thr


def part_a(data, real, S):
    base = {c: st.source_of(c) for c in C.VIOLATIONS}
    cand = dict(base)
    cand["femur_positioning"] = "geo"
    print("(A) парная проверка femur_positioning->geo")
    print(f"    база: {base}")
    print(f"    канд: {cand}\n")
    mb, mc, wins, per_b, per_c = [], [], 0, {}, {}
    for s in range(N_SEEDS):
        idx, splits = _splits(data, real, s)
        fb = np.zeros((len(data), len(C.VIOLATIONS)), dtype=int)
        fc = np.zeros((len(data), len(C.VIOLATIONS)), dtype=int)
        for tr_i, va_i in splits:
            tr, va = idx[tr_i], idx[va_i]
            for tag, sm, out in (("b", base, fb), ("c", cand, fc)):
                thr = _per_criterion_thresholds(data, tr, S, sm)
                f = _fire(data, va, S, sm, thr)
                out |= f
        sb, pb = _macro(data, idx, fb, per_label=True)
        sc, pc = _macro(data, idx, fc, per_label=True)
        mb.append(sb)
        mc.append(sc)
        wins += int(sc > sb)
        for k in pb:
            per_b.setdefault(k, []).append(pb[k])
            per_c.setdefault(k, []).append(pc[k])
    mb, mc = np.array(mb), np.array(mc)
    d = mc - mb
    print(f"    база  macro-F1 {mb.mean():.3f} +- {mb.std():.3f}")
    print(f"    канд  macro-F1 {mc.mean():.3f} +- {mc.std():.3f}")
    print(f"    разница {d.mean():+.3f} +- {d.std():.3f}, выигрыш в {wins}/{N_SEEDS} разбиений")
    print("    по меткам: " + "  ".join(
        f"{k}: {np.mean(per_b[k]):.3f}->{np.mean(per_c[k]):.3f}" for k in per_b))
    print()


def part_b(data, real, S):
    sm = {c: st.source_of(c) for c in C.VIOLATIONS}
    print("(B) совместный подбор порогов под macro-F1 организатора")
    print(f"    конфигурация источников: {sm}, режим базы: {C.THRESHOLD_MODE}\n")
    mi, mj = [], []
    for s in range(N_SEEDS):
        idx, splits = _splits(data, real, s)
        fi = np.zeros((len(data), len(C.VIOLATIONS)), dtype=int)
        fj = np.zeros((len(data), len(C.VIOLATIONS)), dtype=int)
        for tr_i, va_i in splits:
            tr, va = idx[tr_i], idx[va_i]
            thi = _per_criterion_thresholds(data, tr, S, sm)
            thj = _joint_thresholds(data, tr, S, sm)
            fi |= _fire(data, va, S, sm, thi)
            fj |= _fire(data, va, S, sm, thj)
        mi.append(_macro(data, idx, fi))
        mj.append(_macro(data, idx, fj))
    mi, mj = np.array(mi), np.array(mj)
    d = mj - mi
    print(f"    независимые пороги (база) {mi.mean():.3f} +- {mi.std():.3f}")
    print(f"    совместные пороги          {mj.mean():.3f} +- {mj.std():.3f}")
    print(f"    разница {d.mean():+.3f} +- {d.std():.3f}, "
          f"выигрыш в {int((d > 0).sum())}/{N_SEEDS} разбиений")


def main() -> None:
    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    S = E.build_scores(data, real)
    part_a(data, real, S)
    part_b(data, real, S)


if __name__ == "__main__":
    main()
