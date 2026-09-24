# -*- coding: utf-8 -*-
"""Сравнение «до/после» на ОДНОЙ честной схеме.

Что изменилось в пайплайне по итерациям:
  * было — гибрид (CNN + геометрия) для всех критериев + режим prior;
  * итерация 2 — источник по критерию (чистая сеть, гибрид только для оси
    позвоночника) + режим порога f1 (config.CRITERION_SOURCES, THRESHOLD_MODE);
  * итерация 3 — для femur_positioning источник заменён на чистую геометрию.

Изменения отбирались по F1 / честной CV, а не по глобальному AUC.

Запуск:
    cd dxa_qc && python ../scripts/eval_before_after.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dxa_qc"))

from src import config as C        # noqa: E402
from src import dataset as ds      # noqa: E402
from src import features as fl     # noqa: E402
from src import stack as st        # noqa: E402

import eval_honest_organizer as E  # noqa: E402
import eval_final as F             # noqa: E402

N_SEEDS = 20


def main() -> None:
    data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
    real = np.asarray(ds.real_mask(data))
    S = E.build_scores(data, real)
    crits = list(C.VIOLATIONS)
    # конфигурация итерации 2: бедро на CNN (в итерации 3 переведено на геометрию)
    sm_iter2 = {"spine_positioning": "cnn", "spine_axis": "fused",
                "spine_artifacts": "cnn", "femur_positioning": "cnn",
                "femur_roi": "cnn"}

    variants = {
        "БЫЛО: гибрид везде + prior": ({c: "fusedold" for c in crits}, "prior"),
        "итерация 2: источники (бедро=cnn) + f1": (sm_iter2, "f1"),
        "итерация 3 (итог): бедро=геометрия + f1": ({c: st.source_of(c) for c in crits}, "f1"),
    }
    print(f"Честная оценка ({N_SEEDS} разбиений, объединённые val-предсказания):\n")
    print(f"{'вариант':40} {'macro-F1':>16} {'quality BA':>14} {'quality F1':>14}")
    for title, (sm, mode) in variants.items():
        res = [F.run(data, real, S, sm, s, mode) for s in range(N_SEEDS)]
        m = np.array([r["macro"] for r in res])
        ba = np.array([r["ba"] for r in res])
        qf = np.array([r["qf1"] for r in res])
        print(f"{title:40} {m.mean():.3f} ± {m.std():.3f}   "
              f"{ba.mean():.3f} ± {ba.std():.3f}   {qf.mean():.3f} ± {qf.std():.3f}")
    print("\nПрирост (macro-F1):")
    base = np.array([F.run(data, real, S, {c: "fusedold" for c in crits}, s, "prior")["macro"]
                     for s in range(N_SEEDS)])
    fin = np.array([F.run(data, real, S, {c: st.source_of(c) for c in crits}, s, "f1")["macro"]
                    for s in range(N_SEEDS)])
    d = fin - base
    print(f"  {base.mean():.3f} -> {fin.mean():.3f}  (+{d.mean():.3f}, "
          f"улучшение в {int((d > 0).sum())}/{N_SEEDS} разбиениях)")


if __name__ == "__main__":
    main()
