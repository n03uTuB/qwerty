# -*- coding: utf-8 -*-
"""Метрики В ТОЧНОСТИ КАК У ОРГАНИЗАТОРА.

Организатор считает Macro-F1 по 4 уникальным меткам закрытого списка
(``Некорректная укладка`` общая для двух областей). Скрипт воспроизводит
итоговое решение пайплайна (порог по области -> сработавшие критерии) по
OOF-вероятностям и считает:

  * F1 по каждой из 4 меток и их macro-F1 (главная метрика);
  * метрики ``quality_class``: balanced accuracy, macro-F1, ROC-AUC.

Только реальные снимки. Запуск после train + stack:
    cd dxa_qc && python ../scripts/score_organizer.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dxa_qc"))

from src import config as C      # noqa: E402
from src import dataset as ds    # noqa: E402

# 4 метки организатора -> (критерий области, метка области)
ORG_LABELS = {
    "Некорректная укладка": ["spine_positioning", "femur_positioning"],
    "Не выравнена ось позвоночника": ["spine_axis"],
    "Присутствуют посторонние предметы": ["spine_artifacts"],
    "Некорректная область интереса": ["femur_roi"],
}
GT_COL = {
    "spine_positioning": "viol_spine_positioning",
    "spine_axis": "viol_spine_axis",
    "spine_artifacts": "viol_spine_artifacts",
    "femur_positioning": "viol_femur_positioning",
    "femur_roi": "viol_femur_roi",
}


def _decision(df, oof_q, oof_v, thresholds, gate_quality=True):
    """Повторить решение inference.py по OOF: качество по области + критерии.

    gate_quality=True  — как в текущем пайплайне: при quality_class==0 нарушения не
                         выдаются (гейт по качеству).
    gate_quality=False — критерии независимы; quality_class = ИЛИ сработавших
                         критериев (согласованность без потери нарушений).
    """
    qbr = thresholds.get("quality_by_region", {})
    vth = thresholds.get("violations", {})
    q_thr_default = thresholds.get("quality", {}).get("threshold", 0.5)

    quality = np.zeros(len(df), dtype=int)
    fired = {name: np.zeros(len(df), dtype=int) for name in C.VIOLATIONS}
    for i, region in enumerate(df["region"].values):
        q_thr = qbr.get(region, {}).get("threshold", q_thr_default)
        qc = int(oof_q[i] >= q_thr)
        crit = C.REGION_CRITERIA[region]
        got = []
        if gate_quality and qc == 0:
            quality[i] = 0
            continue
        for name in crit:
            thr = vth.get(name, {}).get("threshold", 0.5)
            if oof_v[i, C.VIOLATION_IDX[name]] >= thr:
                got.append(name)
        if gate_quality and got == []:
            # согласованность: при классе «есть нарушение» должен быть указан
            # хотя бы один критерий из закрытого списка
            got = [max(crit, key=lambda n: oof_v[i, C.VIOLATION_IDX[n]])]
        for name in got:
            fired[name][i] = 1
        quality[i] = int(bool(got)) if not gate_quality else qc
    return quality, fired


def _report(df, real, quality, fired, title):
    print(f"\n=== {title} ===")
    print(f"{'метка':34} {'n_pos':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'F1':>6}")
    f1s = []
    for label, crits in ORG_LABELS.items():
        gt = np.zeros(len(df), dtype=int)
        pr = np.zeros(len(df), dtype=int)
        for name in crits:
            col = GT_COL[name]
            if col in df:
                v = df[col].values.astype(float)
                gt |= np.nan_to_num(v, nan=0.0).astype(int)
            pr |= fired[name]
        tp = int(((gt == 1) & (pr == 1) & real).sum())
        fp = int(((gt == 0) & (pr == 1) & real).sum())
        fn = int(((gt == 1) & (pr == 0) & real).sum())
        f1 = f1_score(gt[real], pr[real], zero_division=0)
        f1s.append(f1)
        print(f"{label:34} {int((gt[real] == 1).sum()):6d} {tp:4d} {fp:4d} {fn:4d} {f1:6.3f}")
    print(f"MACRO-F1 (4 метки): {np.mean(f1s):.3f}")

    yq = df["quality"].values.astype(float)
    known = ~np.isnan(yq) & real
    yq_bin = np.nan_to_num(yq, nan=0.0).astype(int)[known]
    pq = quality[known]
    print(f"quality_class: bal.acc={balanced_accuracy_score(yq_bin, pq):.3f}  "
          f"macro-F1={f1_score(yq_bin, pq, average='macro', zero_division=0):.3f}  "
          f"доля(pred/true)={pq.mean():.3f}/{yq_bin.mean():.3f}")
    return float(np.mean(f1s))


def _quality_prob(df, oof_q, oof_v):
    """Вероятность качества ровно как в inference: max по критериям области.

    Именно эта величина (а не «сырая» CNN-голова) идёт в quality_prob пайплайна.
    """
    prob = np.full(len(df), np.nan)
    for i, region in enumerate(df["region"].values):
        crit = C.REGION_CRITERIA[region]
        vals = [oof_v[i, C.VIOLATION_IDX[n]] for n in crit]
        vals = [float(v) for v in vals if not np.isnan(v)]
        prob[i] = max(vals) if vals else float(oof_q[i])
    return prob


def main() -> None:
    manifest = ds.load_manifest(C.MANIFEST_CSV)
    real = np.asarray(ds.real_mask(manifest))

    oof_q = np.load(os.path.join(C.ARTIFACTS_DIR, "oof_quality.npy"))
    oof_v = np.load(os.path.join(C.ARTIFACTS_DIR, "oof_violation.npy"))
    stack_path = os.path.join(C.ARTIFACTS_DIR, "stack_oof_violation.npy")
    if os.path.isfile(stack_path):
        sv = np.load(stack_path)
        if (~np.isnan(sv)).sum() > 0:
            oof_v = sv
            print("[score] нарушения: шкала по источникам из config.CRITERION_SOURCES")
    with open(os.path.join(C.ARTIFACTS_DIR, "thresholds.json"), encoding="utf-8") as f:
        thresholds = json.load(f)

    df = manifest.reset_index(drop=True)
    # OOF-массивы должны соответствовать строкам манифеста. Обучение могло идти без
    # синтетики (252 реальные строки), а манифест уже содержит её (452): генератор
    # дописывает синтетику В КОНЕЦ, поэтому реальные строки идут первыми.
    if len(oof_q) != len(df):
        head = df.iloc[: len(oof_q)]
        if int(head["synthetic"].sum()) == 0 and len(oof_q) == int((~df["synthetic"]).sum()):
            df = head.reset_index(drop=True)
            print(f"[score] OOF согласованы по первым реальным строкам: n={len(df)}")
        else:
            raise SystemExit(
                f"длина OOF ({len(oof_q)}) не совпадает ни с манифестом ({len(manifest)}), "
                f"ни с реальными строками ({int((~manifest['synthetic']).sum())}). "
                f"Перезапустите обучение.")
    real = np.asarray(ds.real_mask(df))
    q_gated, fired_gated = _decision(df, oof_q, oof_v, thresholds, gate_quality=True)
    q_ung, fired_ung = _decision(df, oof_q, oof_v, thresholds, gate_quality=False)
    m_gated = _report(df, real, q_gated, fired_gated, "с гейтом по quality_class (как было)")
    m_ung = _report(df, real, q_ung, fired_ung, "без гейта: quality_class = ИЛИ(нарушений)")
    print(f"\nИТОГ: macro-F1 с гейтом {m_gated:.3f}  |  без гейта {m_ung:.3f}")

    yq = df["quality"].values.astype(float)
    yq_known = ~np.isnan(yq) & real
    yq_bin = np.nan_to_num(yq, nan=0.0).astype(int)[yq_known]
    if len(np.unique(yq_bin)) > 1:
        p_cnn = np.asarray(oof_q)[yq_known]
        p_final = _quality_prob(df, oof_q, oof_v)[yq_known]
        print(f"quality_class ROC-AUC: CNN-голова {roc_auc_score(yq_bin, p_cnn):.3f}  |  "
              f"max(критерии), как в пайплайне {roc_auc_score(yq_bin, p_final):.3f}")


if __name__ == "__main__":
    main()
