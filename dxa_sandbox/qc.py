"""Модель результата контроля качества, типы нарушений и проверки метаданных."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from pydicom.dataset import Dataset

from .reading import pixel_spacing

VIOLATION_SEVERITIES = ("warning", "error")  # info — справочная информация, не нарушение


@dataclass(frozen=True)
class DefectType:
    code: str
    title: str
    # study — дефект выполнения исследования (формулировки п. 3.1 требований ЦДТ),
    # markup — ошибка разметки, data — неполные данные DICOM
    group: str


DEFECTS: dict[str, DefectType] = {
    d.code: d
    for d in [
        DefectType("incomplete_coverage", "Некорректный выбор границ исследования (неполный охват целевого органа)", "study"),
        DefectType("positioning", "Нарушение укладки и позиционирования", "study"),
        DefectType("foreign_body_artifact", "Наличие артефактов от инородных тел", "study"),
        DefectType("acquisition_parameters", "Некорректный выбор физико-технических параметров регистрации изображения", "study"),
        DefectType("vertebra_labeling", "Некорректная нумерация позвонков", "markup"),
        DefectType("roi_placement", "Некорректное положение разметки (границы позвонков, ROI)", "markup"),
        DefectType("dicom_data", "Неполные данные в тегах DICOM", "data"),
    ]
}


@dataclass
class Finding:
    code: str  # машинный код находки, например "spine_axis_tilt"
    title: str  # короткий текст для врача
    severity: str = "warning"  # info | warning | error
    details: str = ""
    value: float | None = None
    unit: str | None = None
    polylines: list[list[tuple[float, float]]] = field(default_factory=list)  # (x, y) в пикселях
    defect: str | None = None  # ключ DEFECTS
    confidence: float = 1.0  # вероятность того, что нарушение есть, 0.00–1.00


@dataclass
class QCResult:
    findings: list[Finding] = field(default_factory=list)

    @property
    def violations(self) -> list[Finding]:
        return [f for f in self.findings if f.severity in VIOLATION_SEVERITIES]

    @property
    def has_violations(self) -> bool:
        return bool(self.violations)

    @property
    def defects(self) -> set[str]:
        return {f.defect for f in self.violations if f.defect}

    @property
    def defect_probability(self) -> float:
        return max((f.confidence for f in self.findings if f.defect), default=0.0)

    @property
    def verdict(self) -> str:
        return "Есть нарушения" if self.has_violations else "Качественное исследование"

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "has_violations": self.has_violations,
            "defects": sorted(self.defects),
            "defect_probability": round(self.defect_probability, 3),
            "findings": [asdict(f) for f in self.findings],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


def body_text(ds: Dataset) -> str:
    return f"{ds.get('BodyPartExamined', '')} {ds.get('SeriesDescription', '')}".upper()


def is_spine(ds: Dataset) -> bool:
    return any(key in body_text(ds) for key in ("SPINE", "L1", "LUMBAR", "ПОЗВОН"))


def is_hip(ds: Dataset) -> bool:
    return any(key in body_text(ds) for key in ("HIP", "FEMUR", "БЕДР"))


def metadata_checks(ds: Dataset) -> list[Finding]:
    """Проверки по тегам. Без них нельзя доверять ни T-score, ни миллиметрам."""
    findings = []
    has_pixels = "PixelData" in ds
    if has_pixels and pixel_spacing(ds) is None:
        findings.append(Finding("no_pixel_spacing", "Нет размера пикселя", "error",
                                "Нельзя перевести расстояния в мм (например, отступ ниже большого вертела)",
                                defect="dicom_data"))
    if not ds.get("PatientSex"):
        findings.append(Finding("missing_patient_sex", "Не указан пол пациента", "warning",
                                "От пола зависит референсная база для T- и Z-score", defect="dicom_data"))
    if not ds.get("PatientAge") and not ds.get("PatientBirthDate"):
        findings.append(Finding("missing_patient_age", "Не указан возраст пациента", "warning",
                                "Возраст нужен для Z-score и выбора критериев интерпретации", defect="dicom_data"))
    if not ds.get("PatientSize") or not ds.get("PatientWeight"):
        findings.append(Finding("missing_height_weight", "Нет роста или веса", "info"))
    if has_pixels and not ds.get("BodyPartExamined") and not ds.get("SeriesDescription"):
        findings.append(Finding("unknown_body_part", "Область исследования не указана в тегах", "warning",
                                "Нужен классификатор по изображению", defect="dicom_data"))
    if is_hip(ds) and not (ds.get("Laterality") or ds.get("ImageLaterality")):
        findings.append(Finding("missing_laterality", "Не указана сторона бедра", "warning", defect="dicom_data"))
    if ds.get("BurnedInAnnotation") == "YES":
        findings.append(Finding("burned_in_annotation", "Разметка или текст вшиты в пиксели", "info"))
    return findings
