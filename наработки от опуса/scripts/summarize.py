# -*- coding: utf-8 -*-
"""Сводка всех экспериментов: собирает out/*.json в единую таблицу и markdown.

Запуск:
    python dxa_qc_work/scripts/summarize.py
"""
from __future__ import annotations

import json
import os

import _common


def load(name):
    p = os.path.join(_common.OUT, name)
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    lines = ["# Сводка экспериментов DXA-QC (честный протокол, 20 разбиений)", ""]

    b = load("exp_baseline.json")
    if b:
        r = b["baseline"]
        lines.append(f"**База (репозиторий):** macro-F1 = {r['macro']:.3f} ± {r['macro_std']:.3f}, "
                     f"BA = {r['ba']:.3f}, AUC = {r['auc']:.3f}")
        lines.append("")

    c = load("exp_confirm.json")
    if c:
        lines += ["## Парное подтверждение кандидатов (v3)", "",
                  "| конфигурация | macro-F1 | Δ | выигрыш |", "|---|---|---|---|",
                  f"| база | {c['base']['mean']:.3f} ± {c['base']['std']:.3f} | — | — |"]
        for k, v in c.items():
            if k == "base" or not isinstance(v, dict) or "wins" not in v:
                continue
            lines.append(f"| {k} | {v['mean']:.3f} ± {v['std']:.3f} | {v['diff']:+.3f} | "
                         f"{v['wins']}/{v['n']} |")
        lines.append("")

    for fname, title in [("exp_featsets.json", "Новые наборы признаков"),
                         ("exp_femur.json", "Фокус на femur_positioning"),
                         ("exp_model_class.json", "Классы моделей"),
                         ("exp_pairs.json", "Прочие кандидаты")]:
        d = load(fname)
        if not d:
            continue
        lines += [f"## {title}", "", "| вариант | macro-F1 | Δ | wins |", "|---|---|---|---|"]
        for k, v in d.items():
            if not isinstance(v, dict) or "mean" not in v:
                continue
            lines.append(f"| {k} | {v['mean']:.3f} ± {v.get('std', 0):.3f} | "
                         f"{v.get('diff', 0):+.3f} | {v.get('wins', '-')} |")
        lines.append("")

    md = "\n".join(lines)
    out = os.path.join(_common.OUT, "SUMMARY.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"\nсохранено: {out}")


if __name__ == "__main__":
    main()
