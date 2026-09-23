# -*- coding: utf-8 -*-
"""Отчёт с метриками: по областям, типам нарушений и с доверительными интервалами.

Читает OOF-предсказания кросс-валидации и манифест, сохраняет
artifacts/report.md и artifacts/metrics_by_region.json.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score,
                             roc_auc_score)

from . import config as C
from . import dataset as ds
from .train import best_threshold, bootstrap_ci, pick_threshold


def quality_block(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    valid = ~np.isnan(y) & ~np.isnan(p)
    y, p = y[valid], p[valid]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return dict(n=int(len(y)), f1=None)
    thr, f1 = pick_threshold(y, p)
    pred = (p >= thr).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    return dict(
        n=int(len(y)), pos=int((y == 1).sum()),
        f1=float(f1), f1_ci=bootstrap_ci(y, p, "f1", thr=thr),
        auc=float(roc_auc_score(y, p)), auc_ci=bootstrap_ci(y, p, "auc"),
        ap=float(average_precision_score(y, p)), ap_ci=bootstrap_ci(y, p, "ap"),
        threshold=float(thr),
        sensitivity=float(tp / max(tp + fn, 1)),
        specificity=float(tn / max(tn + fp, 1)),
        tp=tp, fp=fp, fn=fn, tn=tn,
    )


def fmt_ci(ci):
    if ci is None or any(np.isnan(ci)):
        return "n/a"
    return "[%.3f, %.3f]" % (ci[0], ci[1])


# Соответствие внутренних критериев меткам закрытого списка организатора
_INTERNAL_TO_LABEL = {
    "spine_positioning": "Некорректная укладка",
    "femur_positioning": "Некорректная укладка",
    "spine_axis": "Не выравнена ось позвоночника",
    "spine_artifacts": "Присутствуют посторонние предметы",
    "femur_roi": "Некорректная область интереса",
}


def output_label_matrix(df, prob_v, thresholds, mask=None):
    """Матрицы истина/предсказание (n×4) по закрытому списку организатора.

    Для каждой строки учитываются только критерии её анатомической области.
    ``mask`` ограничивает оценку (синтетику в метрику не берём).
    """
    from sklearn.metrics import f1_score

    labels = C.VIOLATION_LABELS_UNIQUE
    n = len(df)
    keep = np.ones(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    y_true = np.zeros((n, len(labels)), dtype=int)
    y_pred = np.zeros((n, len(labels)), dtype=int)
    for i in range(n):
        region = df["region"].values[i]
        for name in C.REGION_CRITERIA[region]:
            j = C.VIOLATION_IDX[name]
            k = labels.index(_INTERNAL_TO_LABEL[name])
            tv = df["viol_" + name].values[i]
            if not (isinstance(tv, float) and np.isnan(tv)) and float(tv) > 0.5:
                y_true[i, k] = 1
            pv = prob_v[i, j]
            if not np.isnan(pv) and pv >= thresholds.get(name, 0.5):
                y_pred[i, k] = 1
    y_true, y_pred = y_true[keep], y_pred[keep]
    per_label = {}
    f1s = []
    for k, lab in enumerate(labels):
        if y_true[:, k].sum() == 0 and y_pred[:, k].sum() == 0:
            per_label[lab] = None
            continue
        f = float(f1_score(y_true[:, k], y_pred[:, k], zero_division=0))
        per_label[lab] = f
        f1s.append(f)
    macro = float(np.mean(f1s)) if f1s else None
    return per_label, macro


def output_quality_class(df, prob_v, thresholds, mask=None):
    """quality_class по правилу пайплайна: ИЛИ(сработавших критериев области).

    Возвращает массив по всем строкам df (маскирование — на стороне вызова).
    """
    n = len(df)
    q = np.zeros(n, dtype=int)
    for i in range(n):
        for name in C.REGION_CRITERIA[df["region"].values[i]]:
            pv = prob_v[i, C.VIOLATION_IDX[name]]
            if not np.isnan(pv) and pv >= thresholds.get(name, 0.5):
                q[i] = 1
                break
    return q


def main():
    df = ds.load_manifest(C.MANIFEST_CSV)
    oof_q = np.load(os.path.join(C.ARTIFACTS_DIR, "oof_quality.npy"))
    oof_v = np.load(os.path.join(C.ARTIFACTS_DIR, "oof_violation.npy"))
    oof_r = np.load(os.path.join(C.ARTIFACTS_DIR, "oof_region.npy"))

    # нарушения — гибридная шкала стекера (совпадает с инференсом)
    stack_v_path = os.path.join(C.ARTIFACTS_DIR, "stack_oof_violation.npy")
    if os.path.isfile(stack_v_path):
        sv = np.load(stack_v_path)
        if (~np.isnan(sv)).sum() > 0:
            oof_v = sv

    out = {"overall": {}, "by_region": {}, "violations": {}}

    # отчёт строим ТОЛЬКО по реальным снимкам: синтетика участвует лишь в обучении
    real = ds.real_mask(df)
    out["n_synthetic_excluded"] = int((~real).sum())

    # ---- качество: общее и по областям ----
    out["overall"] = quality_block(df["quality"].values[real], oof_q[real])
    for region in C.REGIONS:
        m = (df["region"] == region).values & real
        out["by_region"][region] = quality_block(df["quality"].values[m], oof_q[m])

    # ---- точность определения области ----
    valid_r = ~np.isnan(oof_r[:, 0]) & real
    out["region_accuracy"] = float(
        (oof_r[valid_r].argmax(1) == df["region_idx"].values[valid_r]).mean())

    # ---- нарушения ----
    for i, name in enumerate(C.VIOLATIONS):
        col = "viol_" + name
        y = df[col].values.astype(float)[real]
        p = oof_v[real, i]
        out["violations"][name] = quality_block(y, p)

    # ---- сохранение json ----
    with open(os.path.join(C.ARTIFACTS_DIR, "metrics_by_region.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)

    # ---- markdown ----
    L = []
    L.append("# Метрики DXA-QC (кросс-валидация, GroupKFold по study_uid)\n")
    L.append("## Классификация качества (общая)\n")
    L.append("_Базовая оценка CNN-качества (для сравнения). Итоговое решение: "
             "`quality_class = ИЛИ(сработавших критериев)`._\n")
    o = out["overall"]
    L.append("| Метрика | Значение | 95% ДИ |")
    L.append("|---|---|---|")
    L.append("| N изображений | %d | |" % o["n"])
    L.append("| Доля нарушений | %.3f (%d) | |" % (o["pos"] / o["n"], o["pos"]))
    L.append("| F1 | %.3f | %s |" % (o["f1"], fmt_ci(o["f1_ci"])))
    L.append("| ROC-AUC | %.3f | %s |" % (o["auc"], fmt_ci(o["auc_ci"])))
    L.append("| PR-AUC | %.3f | %s |" % (o["ap"], fmt_ci(o["ap_ci"])))
    L.append("| Чувствительность | %.3f | |" % o["sensitivity"])
    L.append("| Специфичность | %.3f | |" % o["specificity"])
    L.append("| Порог | %.3f | |\n" % o["threshold"])

    L.append("## Качество по анатомическим областям\n")
    L.append("| Область | N | Нарушений | F1 | ROC-AUC | Чувств. | Спец. |")
    L.append("|---|---|---|---|---|---|---|")
    for region, m in out["by_region"].items():
        if m.get("f1") is None:
            L.append("| %s | %d | - | - | - | - | - |" % (region, m["n"]))
            continue
        L.append("| %s | %d | %d | %.3f | %.3f | %.3f | %.3f |" % (
            region, m["n"], m["pos"], m["f1"], m["auc"], m["sensitivity"], m["specificity"]))

    L.append("\n## Типы нарушений (мультилейбл)\n")
    L.append("| Критерий | N | Pos | F1 | ROC-AUC | Порог |")
    L.append("|---|---|---|---|---|---|")
    for name, m in out["violations"].items():
        if m.get("f1") is None:
            L.append("| %s | %d | - | - | - | - |" % (name, m["n"]))
            continue
        L.append("| %s | %d | %d | %.3f | %.3f | %.3f |" % (
            name, m["n"], m["pos"], m["f1"], m["auc"], m["threshold"]))

    L.append("\n## Определение анатомической области\n")
    L.append("Accuracy: **%.4f**\n" % out["region_accuracy"])

    # ---- Macro-F1 по закрытому списку организатора (4 метки) ----
    _thr_path = os.path.join(C.ARTIFACTS_DIR, "thresholds.json")
    _thr_all = {}
    if os.path.isfile(_thr_path):
        with open(_thr_path, encoding="utf-8") as f:
            _thr_all = json.load(f)
    vth_rep = {n: v["threshold"] for n, v in
               _thr_all.get("violations", {}).items()
               if v.get("threshold") is not None}
    per_label, macro = output_label_matrix(df, oof_v, vth_rep, mask=real)
    out["violations_output_labels"] = per_label
    out["violation_macro_f1_output"] = macro

    # ---- метрики ИТОГОВОГО решения пайплайна: quality_class = ИЛИ(критериев) ----
    # Вероятность качества — максимум по критериям области (как в inference).
    q_prob = np.full(len(df), np.nan)
    for i, region in enumerate(df["region"].values):
        vals = [float(oof_v[i, C.VIOLATION_IDX[nm]]) for nm in C.REGION_CRITERIA[region]]
        vals = [v for v in vals if not np.isnan(v)]
        q_prob[i] = max(vals) if vals else float(oof_q[i])
    from sklearn.metrics import balanced_accuracy_score, f1_score as _f1, roc_auc_score
    yq = df["quality"].values.astype(float)
    known = ~np.isnan(yq) & real
    yqb = np.nan_to_num(yq, nan=0.0).astype(int)[known]
    pred_q = output_quality_class(df, oof_v, vth_rep, mask=real)[known]
    pq = q_prob[known]
    ok = ~np.isnan(pq)
    out["decision"] = dict(
        macro_f1_output=macro,
        quality_balanced_acc=float(balanced_accuracy_score(yqb, pred_q)),
        quality_macro_f1=float(_f1(yqb, pred_q, average="macro", zero_division=0)),
        quality_roc_auc=float(roc_auc_score(yqb[ok], pq[ok])) if len(np.unique(yqb[ok])) > 1 else None,
    )
    with open(os.path.join(C.ARTIFACTS_DIR, "metrics_by_region.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    L.append("\n## Итоговое решение (quality_class = ИЛИ(критериев))\n")
    L.append("| Метрика | Значение |")
    L.append("|---|---|")
    L.append("| Macro-F1 (4 метки организатора) | %.3f |" % (macro or 0.0))
    L.append("| Balanced accuracy quality_class | %.3f |" % out["decision"]["quality_balanced_acc"])
    L.append("| Macro-F1 quality_class | %.3f |" % out["decision"]["quality_macro_f1"])
    L.append("| ROC-AUC quality_class (max по критериям) | %.3f |" %
             (out["decision"]["quality_roc_auc"] or 0.0))
    L.append("")
    L.append("\n## Типы нарушений в формате организатора (закрытый список)\n")
    L.append("| Значение violation_type | F1 |")
    L.append("|---|---|")
    for lab, f in per_label.items():
        L.append("| %s | %s |" % (lab, "-" if f is None else "%.3f" % f))
    L.append("| **Macro-F1** | **%s** |" % ("-" if macro is None else "%.3f" % macro))
    L.append("")

    # ---- гибрид (CNN + геометрия), если посчитан ----
    stack_path = os.path.join(C.ARTIFACTS_DIR, "stack_metrics.json")
    if os.path.isfile(stack_path):
        with open(stack_path, encoding="utf-8") as f:
            st = json.load(f)
        comp = st.get("comparison", {})
        L.append("\n## Гибридная модель (CNN + геометрические признаки)\n")
        L.append("| Критерий | CNN F1 | CNN AUC | Гибрид F1 | Гибрид AUC |")
        L.append("|---|---|---|---|---|")

        def _row(title, key):
            a = comp.get(key + "_cnn_only", {})
            b = comp.get(key + "_fused", {})
            if a.get("f1") is None or b.get("f1") is None:
                L.append("| %s | - | - | - | - |" % title)
            else:
                L.append("| %s | %.3f | %.3f | %.3f | %.3f |" % (
                    title, a["f1"], a["auc"], b["f1"], b["auc"]))

        for name in C.VIOLATIONS:
            _row(name, name)
        L.append("")
        L.append("Геометрические признаки применяются к критериям нарушений; "
                 "класс качества = ИЛИ(сработавших критериев) — гейт по "
                 "нейросети снят (см. docs/metrics_improvement.md).\n")

    path = os.path.join(C.ARTIFACTS_DIR, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("[report] сохранено:", path)
    print("\n".join(L))

    # ---- сохранить OOF-предсказания в формате, близком к ТЗ ----
    from . import inference as _inf

    thr = _inf.load_thresholds()
    vth = {n: v["threshold"] for n, v in thr.get("violations", {}).items()
           if v.get("threshold") is not None}
    oof_rows = []
    for i, row in df.iterrows():
        region = row["region"]
        qp = oof_q[i]
        if np.isnan(qp):
            continue
        fired = []
        crit = C.REGION_CRITERIA[region]
        max_viol = 0.0
        for name in crit:
            j = C.VIOLATION_IDX[name]
            p = oof_v[i, j]
            if not np.isnan(p):
                max_viol = max(max_viol, float(p))
                if p >= vth.get(name, 0.5):
                    fired.append(name)
        # та же логика, что в inference.predict_array: quality_class = ИЛИ(критериев)
        qclass = int(bool(fired))
        qprob = max_viol if crit else float(qp)
        oof_rows.append(dict(
            path_to_study=row["source_path"], study_uid=row["study_uid"],
            image_uid=row["image_uid"],
            anatomical_region=_inf.region_label_ru(region),
            quality_class=qclass,
            violation_type=_inf.violation_text_ru(fired),
            processing_status="Success", quality_prob=round(qprob, 4),
            true_quality=row["quality"],
        ))
    oof_df = pd.DataFrame(oof_rows)
    os.makedirs(C.OUTPUTS_DIR, exist_ok=True)
    oof_path = os.path.join(C.OUTPUTS_DIR, "validation_oof.csv")
    oof_df.to_csv(oof_path, index=False, encoding="utf-8-sig")
    print("[report] OOF-предсказания:", oof_path, "строк:", len(oof_df))


if __name__ == "__main__":
    main()
