"""Сквозной пайплайн: исследование → проверки → результат по требованиям ЦДТ.

Реальные проверки и модели встраиваются в run_qc (и checks.image_checks), остальное менять не нужно.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset

from .checks import _rectangle, image_checks
from .compliance import check_outputs
from .config import DEFAULT, ServiceConfig, series_uid_by_mask
from .markup import color_markup_masks, components, overlay_groups, overlay_mask, remove_markup
from .qc import Finding, QCResult, is_hip, is_spine, metadata_checks
from .reading import iter_dicom_files, luminance, read, save_png, to_analysis, to_display
from .writers import (conclusion_text, create_markup_sc, create_qc_sr, description_text, draw_findings,
                      error_message, report_notify_message)

SUPPORTED_MODALITIES = {"BMD", "DX", "CR", "OT"}
NON_IMAGE_MODALITIES = {"SR", "DOC", "PR", "KO"}


class ProcessingError(Exception):
    """Исследование нельзя обработать. category — из таблицы 1 п. 3.5 требований ЦДТ:
    Server unavailable, Incorrect number of images, Modality error, Series error, Tag error,
    Body part error, Images error, Other."""

    def __init__(self, category: str, description: str):
        super().__init__(description)
        self.category = category
        self.description = description


def _dump(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def prepare_image(ds: Dataset) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Картинка для рисования (uint8), значения для проверок (float) и маски цветной разметки."""
    display = to_display(ds)
    if display.ndim == 3:
        masks = color_markup_masks(display)
        if masks:
            values = remove_markup(display, np.logical_or.reduce(list(masks.values()))).astype(np.float32)
        else:
            values = luminance(display)
        return display, values, masks
    return display, to_analysis(ds), {}


def run_qc(ds: Dataset) -> QCResult:
    result = QCResult(metadata_checks(ds))
    if "PixelData" not in ds or ds.get("Modality") in NON_IMAGE_MODALITIES:
        return result
    _, values, masks = prepare_image(ds)
    if masks:
        result.findings.append(Finding("burned_in_markup_found", f"Разметка в пикселях: {', '.join(masks)}", "info"))
    for group in overlay_groups(ds):
        found = components(overlay_mask(ds, group))
        result.findings.append(Finding("overlay_markup_found", f"Разметка в overlay {group:04X}: объектов {len(found)}",
                                       "info", polylines=[_rectangle(c["bbox"]) for c in found]))
    kind = "spine" if is_spine(ds) else "hip" if is_hip(ds) else None
    result.findings.extend(image_checks(values, masks, kind))
    return result


def select_images(datasets: list[tuple[Path, Dataset]]) -> list[tuple[Path, Dataset]]:
    images = [(p, ds) for p, ds in datasets if "PixelData" in ds and ds.get("Modality") not in NON_IMAGE_MODALITIES]
    modalities = {str(ds.get("Modality", "")) for _, ds in images}
    if images and not modalities & SUPPORTED_MODALITIES:
        raise ProcessingError("Modality error", f"Значение DICOM-тега Modality (0008,0060) не поддерживается: "
                                                f"{', '.join(sorted(modalities))}")
    if not images:
        raise ProcessingError("Series error", "В исследовании нет серий, подходящих для обработки")
    return [(p, ds) for p, ds in images if str(ds.get("Modality", "")) in SUPPORTED_MODALITIES]


def _process_images(images, out: Path, config: ServiceConfig, times: dict) -> list[Path]:
    processed_at = datetime.now()
    first = images[0][1]
    sc_series = series_uid_by_mask(str(first.SeriesInstanceUID), config.model_id, add_id=1)
    sr_series = series_uid_by_mask(str(first.SeriesInstanceUID), config.model_id, add_id=2)
    study_result, image_results, report_images, sc_paths = QCResult(), [], [], []

    for number, (path, ds) in enumerate(images, start=1):
        try:
            result = run_qc(ds)
            display = to_display(ds)
        except Exception as exc:
            raise ProcessingError("Images error", f"Не удалось обработать изображение {path.name}: {exc}") from exc
        label = str(ds.get("SeriesDescription") or path.stem)
        image_results.append((label, result))
        study_result.findings.extend(replace(f, title=f"{label}: {f.title}", polylines=[]) for f in result.findings)
        report_images.append({"file": path.name, "series_description": label, **result.to_dict()})

        picture = draw_findings(display, result, config, processed_at)
        save_png(picture, out / f"{path.stem}_qc.png")
        sc = create_markup_sc(ds, picture, series_instance_uid=sc_series, instance_number=number,
                              config=config, processed_at=processed_at)
        sc_path = out / f"{path.stem}_qc_sc.dcm"
        sc.save_as(sc_path, enforce_file_format=True)
        sc_paths.append(sc_path)

    sr = create_qc_sr([ds for _, ds in images], study_result, image_results,
                      series_instance_uid=sr_series, config=config, processed_at=processed_at)
    sr_path = out / "qc_report_sr.dcm"
    sr.save_as(sr_path, enforce_file_format=True)
    times["processEndDT"] = datetime.now()

    study_uid = str(first.StudyInstanceUID)
    _dump(out / "dicom_report_notify.json", report_notify_message(
        study_uid, sc_series, study_result, config, times, description_text(image_results), conclusion_text(study_result)))
    _dump(out / "qc_report.json", {
        "study_uid": study_uid,
        "verdict": study_result.verdict,
        "has_violations": study_result.has_violations,
        "defects": sorted(study_result.defects),
        "defect_probability": round(study_result.defect_probability, 3),
        "processing_seconds": round((times["processEndDT"] - times["downloadStartDT"]).total_seconds(), 2),
        "images": report_images,
        "compliance": check_outputs(images, sc_paths, sr_path, config),
    })
    return sc_paths + [sr_path]


def process_study(paths: list[Path], out_dir: str | Path, config: ServiceConfig = DEFAULT) -> list[Path]:
    """Одно исследование → PNG, доп. серия SC, SR, dicom_report_notify.json, qc_report.json.
    Если обработать нельзя — pum_consumer_error.json (прил. 5)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    times = {"downloadStartDT": datetime.now()}
    datasets = []
    for path in paths:
        try:
            datasets.append((Path(path), read(path)))
        except Exception:
            continue
    times["downloadEndDT"] = times["processStartDT"] = datetime.now()
    study_uid = str(datasets[0][1].get("StudyInstanceUID", "")) if datasets else ""
    try:
        if not datasets:
            raise ProcessingError("Images error", "Не удалось прочитать файлы DICOM")
        return _process_images(select_images(datasets), out, config, times)
    except ProcessingError as err:
        _dump(out / "pum_consumer_error.json", error_message(study_uid, err.category, err.description, config, times))
        _dump(out / "qc_report.json", {"study_uid": study_uid,
                                       "error": {"category": err.category, "description": err.description}})
        return []


def process_path(path: str | Path, out_dir: str | Path, config: ServiceConfig = DEFAULT) -> dict[str, list[Path]]:
    """Файл или папка: группирует файлы по исследованиям и обрабатывает каждое."""
    by_study = defaultdict(list)
    for file in iter_dicom_files(path):
        header = read(file, pixels=False)
        by_study[str(header.get("StudyInstanceUID", "unknown"))].append(file)
    return {study: process_study(files, Path(out_dir) / study[-12:], config) for study, files in by_study.items()}
