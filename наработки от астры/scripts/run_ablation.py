# -*- coding: utf-8 -*-
"""Честное сравнение «до/после»: базовые наборы vs новые art_*/atlas_*.

Запускать ПОСЛЕ `python -m src.train` (нужны OOF-вероятности CNN).
Для каждого режима (base/new) прогоняет stack -> calibrate -> evaluate и
сравнивает метрики по критериям и метрику организатора.

    python scripts/run_ablation.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import config as C  # noqa: E402
from src.evaluate import EVAL_JSON  # noqa: E402


def run(mode):
    env = dict(os.environ)
    env["DXA_FEATURE_MODE"] = mode
    for mod in (["-m", "src.stack"], ["-m", "src.calibrate"],
                ["-m", "src.evaluate", "--thr", "f1", "--stacked"]):
        r = subprocess.run([sys.executable] + mod, cwd=ROOT, env=env,
                           capture_output=True)
        if r.returncode != 0:
            print(r.stdout.decode("utf-8", "replace")[-3000:])
            print(r.stderr.decode("utf-8", "replace")[-3000:])
            raise SystemExit("FAILED %s mode=%s" % (mod, mode))
    dst = os.path.join(C.ARTIFACTS_DIR, "evaluate_%s.json" % mode)
    shutil.copy(EVAL_JSON, dst)
    return json.load(open(dst, encoding="utf-8"))


def main():
    base = run("base")
    new = run("new")
    lines = []
    lines.append("| критерий | AUC base | AUC new | F1 base | F1 new |")
    lines.append("|---|---|---|---|---|")
    for name in C.VIOLATIONS:
        b = base["violations"].get(name, {})
        n = new["violations"].get(name, {})
        def f(v):
            return "-" if v is None else "%.3f" % v
        lines.append("| %s | %s | %s | %s | %s |" % (
            name, f(b.get("auc")), f(n.get("auc")), f(b.get("f1")), f(n.get("f1"))))
    lines.append("")
    lines.append("organizer macro-F1: base=%.3f -> new=%.3f" % (
        base["organizer"]["macro"], new["organizer"]["macro"]))
    for lbl in base["organizer"]["per_label"]:
        lines.append("  %s: base=%.3f new=%.3f" % (
            lbl, base["organizer"]["per_label"][lbl],
            new["organizer"]["per_label"].get(lbl, float("nan"))))
    qb, qn = base.get("quality_class", {}), new.get("quality_class", {})
    lines.append("quality_class: BA %.3f->%.3f  macroF1 %.3f->%.3f  AUC %.3f->%.3f" % (
        qb.get("balanced_accuracy", 0), qn.get("balanced_accuracy", 0),
        qb.get("macro_f1", 0), qn.get("macro_f1", 0),
        qb.get("auc", 0), qn.get("auc", 0)))
    txt = "\n".join(lines)
    open(os.path.join(ROOT, "ablation_result.md"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()