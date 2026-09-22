"""Метрики проверок на размеченном наборе (например, синтетическом из synthetic.py)."""

from __future__ import annotations

import json
from pathlib import Path

from .qc import DEFECTS

EVAL_DEFECTS = [code for code, defect in DEFECTS.items() if defect.group != "data"]


def load_labels(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def collect_predictions(qc_dir: str | Path) -> dict[str, dict]:
    predictions = {}
    for report_path in Path(qc_dir).rglob("qc_report.json"):
        data = json.loads(report_path.read_text(encoding="utf-8"))
        predictions[data.get("study_uid", "")] = {
            "defects": {d for d in data.get("defects", []) if d in EVAL_DEFECTS},
            "error": data.get("error"),
        }
    return predictions


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 3) if denominator else None


def _binary(pairs: list[tuple[bool, bool]]) -> dict:
    tp = sum(t and p for t, p in pairs)
    fp = sum(p and not t for t, p in pairs)
    fn = sum(t and not p for t, p in pairs)
    tn = sum(not t and not p for t, p in pairs)
    return {"support": tp + fn, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity": _ratio(tp, tp + fn), "specificity": _ratio(tn, tn + fp),
            "precision": _ratio(tp, tp + fp), "f1": _ratio(2 * tp, 2 * tp + fp + fn)}


def evaluate(labels: list[dict], predictions: dict[str, dict]) -> dict:
    predicted = {label["study_uid"]: predictions.get(label["study_uid"], {}).get("defects", set()) for label in labels}
    per_defect = {}
    for code in EVAL_DEFECTS:
        pairs = [(code in label["defects"], code in predicted[label["study_uid"]]) for label in labels]
        per_defect[code] = {"title": DEFECTS[code].title, **_binary(pairs)}
    study = _binary([(bool(label["defects"]), bool(predicted[label["study_uid"]])) for label in labels])
    exact = sum(set(label["defects"]) == predicted[label["study_uid"]] for label in labels)
    return {
        "cases": len(labels),
        "missing_predictions": sum(label["study_uid"] not in predictions for label in labels),
        "study_level": study,
        "exact_match": _ratio(exact, len(labels)),
        "per_defect": per_defect,
    }


def format_report(metrics: dict) -> str:
    def f(value):
        return "  —  " if value is None else f"{value:5.2f}"

    study = metrics["study_level"]
    lines = [
        f"Случаев: {metrics['cases']}, без результата: {metrics['missing_predictions']}",
        f"Есть ли нарушение (уровень исследования): чувствительность {f(study['sensitivity'])}, "
        f"специфичность {f(study['specificity'])}, F1 {f(study['f1'])}",
        f"Точное совпадение набора нарушений: {f(metrics['exact_match'])}",
        "",
        f"{'нарушение':<24}{'случаев':>8}{'чувств.':>9}{'специф.':>9}{'precision':>10}{'F1':>7}",
    ]
    for code, row in metrics["per_defect"].items():
        lines.append(f"{code:<24}{row['support']:>8}{f(row['sensitivity']):>9}{f(row['specificity']):>9}"
                     f"{f(row['precision']):>10}{f(row['f1']):>7}")
    return "\n".join(lines)
