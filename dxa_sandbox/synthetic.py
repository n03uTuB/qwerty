"""Генератор синтетических нарушений для обучения и оценки проверок.

Операции работают с парой «изображение + геометрия» (numpy и координаты), поэтому их можно
применять и к реальным снимкам с экспертной разметкой: сдвиг нумерации, смещение границы,
поворот ROI, поворот снимка (укладка), обрезка кадра, металлический артефакт, шум и низкий контраст.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid

from .geometry import image_center, rotate_image, rotate_points
from .samples import SECONDARY_CAPTURE, VERTEBRAE, _new_dataset, _save, hip_phantom, spine_phantom

Point = tuple[float, float]

APPLICABLE = {
    "spine": ["incomplete_coverage", "positioning", "foreign_body_artifact", "acquisition_parameters",
              "vertebra_labeling", "roi_placement"],
    "hip": ["positioning", "foreign_body_artifact", "acquisition_parameters", "roi_placement"],
}

COLORS = {"boundary": (0, 230, 0), "label": (255, 230, 0), "roi": (230, 0, 0), "total_hip": (0, 230, 0)}


@dataclass
class Case:
    kind: str  # "spine" | "hip"
    image: np.ndarray  # float32 в единицах аппарата (0–4095)
    anatomy: dict[str, list[Point]]  # истинная геометрия: позвонки или экспертные ROI
    markup: dict  # разметка «аппарата»: boundaries, labels, roi, total_hip
    side: str = ""
    defects: list[dict] = field(default_factory=list)  # {"code": ..., параметры}


# ---------------------------------------------------------------- разметка


def _bbox_polygon(polygons, margin: float) -> list[Point]:
    xs = [x for poly in polygons for x, _ in poly]
    ys = [y for poly in polygons for _, y in poly]
    x0, y0, x1, y1 = min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def spine_markup(vertebrae: dict[str, list[Point]], first: int = 1) -> dict:
    """Разметка как у аппарата: границы между позвонками, подписи L1–L4 и ROI L1–L4.
    first — индекс в VERTEBRAE позвонка, подписанного как L1 (1 — правильно)."""
    boundaries = []
    for upper, lower in zip(VERTEBRAE[:-1], VERTEBRAE[1:]):
        u, l = vertebrae[upper], vertebrae[lower]
        left = ((u[3][0] + l[0][0]) / 2 - 10, (u[3][1] + l[0][1]) / 2)
        right = ((u[2][0] + l[1][0]) / 2 + 10, (u[2][1] + l[1][1]) / 2)
        boundaries.append([left, right])
    targets = VERTEBRAE[first : first + 4]
    labels = {
        f"L{i + 1}": ((vertebrae[name][1][0] + vertebrae[name][2][0]) / 2 + 14,
                      (vertebrae[name][1][1] + vertebrae[name][2][1]) / 2 - 6)
        for i, name in enumerate(targets)
    }
    return {"boundaries": boundaries, "labels": labels, "roi": _bbox_polygon([vertebrae[n] for n in targets], 4)}


def _map_points(case: Case, fn) -> None:
    case.anatomy = {k: fn(v) for k, v in case.anatomy.items()}
    markup = case.markup
    markup["boundaries"] = [fn(line) for line in markup.get("boundaries", [])]
    markup["labels"] = {k: fn([p])[0] for k, p in markup.get("labels", {}).items()}
    for key in ("roi", "total_hip"):
        if markup.get(key):
            markup[key] = fn(markup[key])


# ---------------------------------------------------------------- нарушения


def shift_vertebra_labels(case: Case, levels: int) -> None:
    """Ошибка нумерации: подписи и ROI L1–L4 стоят на уровень выше (−1) или ниже (+1)."""
    shifted = spine_markup(case.anatomy, first=1 + levels)
    case.markup["labels"], case.markup["roi"] = shifted["labels"], shifted["roi"]
    case.defects.append({"code": "vertebra_labeling", "levels": levels})


def displace_boundary(case: Case, index: int, dy: float) -> None:
    """Межпозвонковая граница уехала в тело позвонка."""
    case.markup["boundaries"][index] = [(x, y + dy) for x, y in case.markup["boundaries"][index]]
    case.defects.append({"code": "roi_placement", "boundary": index, "dy": round(dy, 1)})


def rotate_polygon(polygon: list[Point], angle_deg: float) -> list[Point]:
    cx = sum(x for x, _ in polygon) / len(polygon)
    cy = sum(y for _, y in polygon) / len(polygon)
    return rotate_points(polygon, angle_deg, (cx, cy))


def rotate_case(case: Case, angle_deg: float) -> None:
    """Поворот снимка вместе со всей геометрией (разметка аппарата следует за пациентом)."""
    case.image = rotate_image(case.image, angle_deg)
    center = image_center(case.image.shape)
    _map_points(case, lambda points: rotate_points(points, angle_deg, center))


def crop_case(case: Case, top: int = 0, bottom: int = 0) -> None:
    rows = case.image.shape[0]
    case.image = case.image[top : rows - bottom]
    _map_points(case, lambda points: [(x, y - top) for x, y in points])
    case.defects.append({"code": "incomplete_coverage", "top": top, "bottom": bottom})


def add_metal(case: Case, rng: np.random.Generator) -> None:
    rows, cols = case.image.shape
    cx, cy = rng.uniform(0.2, 0.8) * cols, rng.uniform(0.15, 0.85) * rows
    shape = str(rng.choice(["disc", "clasp"]))
    image = Image.new("L", (cols, rows), 0)
    draw = ImageDraw.Draw(image)
    if shape == "disc":
        r = rng.uniform(4, 8)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=1)
    else:
        draw.rectangle([cx - 9, cy - 3, cx + 9, cy + 3], fill=1)
    case.image = np.where(np.array(image) > 0, 4000.0, case.image).astype(np.float32)
    case.defects.append({"code": "foreign_body_artifact", "shape": shape, "x": round(cx, 1), "y": round(cy, 1)})


def degrade_acquisition(case: Case, rng: np.random.Generator) -> None:
    """Низкий контраст и сильный шум — как при неверных параметрах съёмки."""
    median = float(np.median(case.image))
    noisy = median + (case.image - median) * 0.5 + rng.normal(0, 150, case.image.shape)
    case.image = np.clip(noisy, 0, 4095).astype(np.float32)
    case.defects.append({"code": "acquisition_parameters", "contrast": 0.5, "noise": 150})


# ---------------------------------------------------------------- сборка случая


def generate_case(kind: str, rng: np.random.Generator, defects: list[str]) -> Case:
    seed = int(rng.integers(1 << 31))
    if kind == "spine":
        image, vertebrae = spine_phantom(int(rng.integers(380, 441)), int(rng.integers(280, 321)), seed=seed)
        anatomy = {k: [tuple(map(float, p)) for p in v] for k, v in vertebrae.items()}
        case = Case("spine", image.astype(np.float32), anatomy, spine_markup(anatomy))
        if "vertebra_labeling" in defects:
            shift_vertebra_labels(case, int(rng.choice([-1, 1])))
        if "roi_placement" in defects:
            displace_boundary(case, int(rng.integers(0, 5)), float(rng.choice([-1, 1]) * rng.uniform(16, 24)))
        if "positioning" in defects:
            angle = float(rng.choice([-1, 1]) * rng.uniform(8, 15))
            rotate_case(case, angle)
            case.defects.append({"code": "positioning", "tilt_deg": round(angle, 1)})
        else:
            rotate_case(case, float(rng.uniform(-2, 2)))
        if "incomplete_coverage" in defects:
            rows = case.image.shape[0]
            if rng.random() < 0.5:
                name = str(rng.choice(["T12", "L1"]))
                crop_case(case, top=int(np.mean([y for _, y in case.anatomy[name]])))
            else:
                name = str(rng.choice(["L4", "L5"]))
                crop_case(case, bottom=rows - int(np.mean([y for _, y in case.anatomy[name]])))
    else:
        side = str(rng.choice(["L", "R"]))
        rotation = float(rng.uniform(25, 35)) if "positioning" in defects else float(rng.uniform(0, 5))
        image, rois = hip_phantom(int(rng.integers(340, 381)), int(rng.integers(280, 321)), side=side,
                                  external_rotation_deg=rotation, seed=seed)
        case = Case("hip", image.astype(np.float32), {k: list(v) for k, v in rois.items()},
                    {"roi": list(rois["neck_roi"]), "total_hip": list(rois["total_hip"])}, side=side)
        if "positioning" in defects:
            case.defects.append({"code": "positioning", "external_rotation_deg": round(rotation, 1)})
        if "roi_placement" in defects:
            angle = float(rng.choice([-1, 1]) * rng.uniform(25, 40))
            case.markup["roi"] = rotate_polygon(case.markup["roi"], angle)
            case.defects.append({"code": "roi_placement", "neck_roi_rotation_deg": round(angle, 1)})
    if "acquisition_parameters" in defects:
        degrade_acquisition(case, rng)
    if "foreign_body_artifact" in defects:  # после шума, чтобы металл не «разбавлялся»
        add_metal(case, rng)
    return case


def render_case(case: Case) -> np.ndarray:
    """Скриншот как у аппарата: снимок с цветной разметкой, вшитой в пиксели."""
    gray = np.clip((case.image - 300) / 3000 * 255, 0, 255).astype(np.uint8)
    picture = Image.fromarray(np.stack([gray] * 3, axis=-1))
    draw = ImageDraw.Draw(picture)
    markup = case.markup
    for line in markup.get("boundaries", []):
        draw.line(line, fill=COLORS["boundary"], width=2)
    for name, (x, y) in markup.get("labels", {}).items():
        draw.text((x, y), name, fill=COLORS["label"])
    for key in ("roi", "total_hip"):
        if markup.get(key):
            draw.line(markup[key] + [markup[key][0]], fill=COLORS[key], width=2)
    return np.array(picture)


def write_case(case: Case, rgb: np.ndarray, path: Path, case_id: str) -> Dataset:
    if case.kind == "spine":
        description, body = "AP Spine L1-L4", {"BodyPartExamined": "LSPINE"}
    else:
        description = "Left Femur" if case.side == "L" else "Right Femur"
        body = {"BodyPartExamined": "HIP", "Laterality": case.side}
    ds = _new_dataset(SECONDARY_CAPTURE, generate_uid(), series_number=1, modality="OT",
                      series_description=description, PatientSex="F", PatientAge="065Y",
                      PatientSize="1.62", PatientWeight="60", PixelSpacing=["0.5", "0.5"],
                      BurnedInAnnotation="YES", ConversionType="WSD", **body)
    ds.PatientID = case_id
    ds.set_pixel_data(rgb, photometric_interpretation="RGB", bits_stored=8)
    _save(ds, path)
    return ds


def _rounded(value):
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(v) for v in value]
    return round(float(value), 1) if isinstance(value, (float, np.floating)) else value


def generate_dataset(out_dir: str | Path, n: int = 100, seed: int = 0, clean_fraction: float = 0.35,
                     two_defects_fraction: float = 0.1, spine_share: float = 0.6) -> list[dict]:
    """cases/*.dcm + labels.jsonl (нарушения, параметры, истинная геометрия и разметка аппарата)."""
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    (out / "cases").mkdir(parents=True, exist_ok=True)
    labels = []
    for i in range(n):
        kind = "spine" if rng.random() < spine_share else "hip"
        if rng.random() < clean_fraction:
            chosen = []
        else:
            k = 2 if rng.random() < two_defects_fraction else 1
            chosen = sorted(rng.choice(APPLICABLE[kind], size=k, replace=False).tolist())
        case = generate_case(kind, rng, chosen)
        case_id = f"case_{i:04d}"
        ds = write_case(case, render_case(case), out / "cases" / f"{case_id}.dcm", case_id)
        labels.append({
            "case_id": case_id, "file": f"cases/{case_id}.dcm", "study_uid": str(ds.StudyInstanceUID),
            "kind": kind, "defects": chosen, "params": _rounded(case.defects),
            "geometry": {"anatomy": _rounded(case.anatomy), "markup": _rounded(case.markup)},
        })
    with open(out / "labels.jsonl", "w", encoding="utf-8") as f:
        for label in labels:
            f.write(json.dumps(label, ensure_ascii=False) + "\n")
    counts = Counter(code for label in labels for code in label["defects"])
    (out / "README.md").write_text(
        "# Синтетический набор (фантом, не медицинские данные)\n\n"
        f"Случаев: {n}, без нарушений: {sum(1 for l in labels if not l['defects'])}, seed={seed}\n\n"
        + "\n".join(f"- {code}: {count}" for code, count in sorted(counts.items()))
        + "\n\n`labels.jsonl`: нарушения, параметры, истинная геометрия (`anatomy`) и разметка аппарата (`markup`).\n",
        encoding="utf-8",
    )
    return labels
