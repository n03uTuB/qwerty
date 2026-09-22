"""Результат по «Базовым функциональным требованиям» ЦДТ (версия от 15.09.2026):
- дополнительная серия Secondary Capture с вшитыми надписями (п. 3.4, прил. 7);
- DICOM SR с фиксированным порядком полей (п. 3.3);
- сообщения о результате и об ошибке в формате прил. 4 и 5 (JSON вместо Kafka).
"""

from __future__ import annotations

from datetime import datetime

import highdicom as hd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydicom.dataset import Dataset

from .config import ServiceConfig
from .qc import DEFECTS, QCResult
from .reading import pixel_spacing

UTF8 = "ISO_IR 192"  # без этого кириллица в тегах сломается
SEVERITY_COLORS = {"info": (80, 170, 255), "warning": (255, 190, 0), "error": (255, 60, 60)}
# п. 3.4 требует «Целевая патология не выявлена»; для сервиса контроля качества целевое — дефекты. Согласовать.
NO_DEFECTS_TEXT = "Технологические дефекты не выявлены"

QC_SCHEME = "99DXAQC"  # локальная схема кодов, согласовать с заказчиком
REPORT_TITLE = hd.sr.CodedConcept(value="18748-4", scheme_designator="LN", meaning="Diagnostic imaging report")

# п. 3.3: структура протокола едина, порядок полей менять нельзя
SR_FIELDS = [
    ("modality", "Модальность"),
    ("body-part", "Область исследования"),
    ("study-uid", "Идентификатор исследования"),
    ("report-datetime", "Дата и время формирования заключения ИИ-сервисом"),
    ("warning-origin", "Предупреждение"),
    ("warning-usage", "Предупреждение"),
    ("service-name", "Наименование сервиса"),
    ("service-version", "Версия сервиса"),
    ("service-purpose", "Назначение сервиса"),
    ("technical-data", "Технические данные"),
    ("technological-defects", "Технологические дефекты при выполнении исследования"),
    ("description", "Описание"),
    ("conclusion", "Заключение"),
    ("user-guide", "Руководство пользователя"),
]

USER_GUIDE = (
    "Дополнительная серия: красный — ошибка, жёлтый — предупреждение, синий — справочная информация. "
    "Линии и рамки указывают место нарушения: ось позвоночника, границы позвонков, ROI, инородное тело. "
    "Классификация дефектов выполнения исследования — по п. 3.1 Базовых функциональных требований ЦДТ; "
    "вероятность нарушения указана в диапазоне 0,00–1,00."
)

BODY_PARTS = {"LSPINE": "поясничный отдел позвоночника", "SPINE": "позвоночник", "HIP": "проксимальный отдел бедра"}
SIDES = {"L": "левое", "R": "правое"}


def _code(value: str, meaning: str) -> hd.sr.CodedConcept:
    return hd.sr.CodedConcept(value=value, scheme_designator=QC_SCHEME, meaning=meaning[:64])


def _font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------- тексты


def body_parts_text(datasets: list[Dataset]) -> str:
    parts = []
    for ds in datasets:
        body = str(ds.get("BodyPartExamined", "") or "")
        text = BODY_PARTS.get(body.upper(), body or str(ds.get("SeriesDescription", "") or "не указана"))
        side = SIDES.get(str(ds.get("Laterality") or ds.get("ImageLaterality") or ""))
        parts.append(f"{text} ({side})" if side else text)
    return "; ".join(dict.fromkeys(parts))


def technical_text(datasets: list[Dataset]) -> str:
    sizes = "; ".join(
        f"{ds.get('SeriesDescription', '')}: {ds.get('Rows')}×{ds.get('Columns')}"
        + (f", пиксель {pixel_spacing(ds).row_mm:g}×{pixel_spacing(ds).col_mm:g} мм" if pixel_spacing(ds) else "")
        for ds in datasets
    )
    return f"Обработано изображений: {len(datasets)}. {sizes}"


def defects_text(result: QCResult) -> str:
    titles = [DEFECTS[c].title for c in DEFECTS if c in result.defects and DEFECTS[c].group == "study"]
    return "; ".join(titles) if titles else "Не выявлены"


def description_text(image_results: list[tuple[str, QCResult]]) -> str:
    lines = []
    for label, result in image_results:
        items = [f"{f.title} (вероятность {f.confidence:.2f})" + (f" — {f.details}" if f.details else "")
                 for f in result.violations]
        lines.append(f"{label}: " + ("; ".join(items) if items else "нарушений не выявлено"))
    return "\n".join(lines)


def conclusion_text(result: QCResult) -> str:
    if not result.has_violations:
        return "Технологические дефекты выполнения исследования и ошибки разметки не выявлены."
    titles = [DEFECTS[c].title for c in DEFECTS if c in result.defects]
    other = [f.title for f in result.violations if not f.defect]
    return "Выявлены нарушения: " + "; ".join(titles + other) + "."


# ---------------------------------------------------------------- дополнительная серия


def draw_findings(image: np.ndarray, result: QCResult, config: ServiceConfig, processed_at: datetime) -> np.ndarray:
    """RGB того же размера: геометрия находок, плашка с вердиктом и неотключаемые надписи п. 3.4."""
    rgb = np.stack([image] * 3, axis=-1) if image.ndim == 2 else image[..., :3]
    canvas = Image.fromarray(rgb.astype(np.uint8)).convert("RGBA")
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    width, height = canvas.size
    line_width = max(1, width // 250)

    for finding in result.findings:
        color = SEVERITY_COLORS.get(finding.severity, (255, 255, 255)) + (255,)
        for line in finding.polylines:
            if len(line) >= 2:
                draw.line([(float(x), float(y)) for x, y in line], fill=color, width=line_width)

    size = max(11, width // 32)
    font = _font(size)
    pad, line_height = 4, size + 3

    header = ["Выявлены нарушения"] + [f"• {f.title}" for f in result.violations] if result.has_violations else [NO_DEFECTS_TEXT]
    draw.rectangle([0, 0, width, pad * 2 + line_height * len(header)], fill=(0, 0, 0, 170))
    for i, text in enumerate(header):
        color = ((255, 90, 90) if result.has_violations else (120, 255, 120)) if i == 0 else (255, 255, 255)
        draw.text((pad, pad + i * line_height), text, font=font, fill=color + (255,))

    footer = [config.image_warning, f"{config.name} v{config.version}", processed_at.strftime("%d.%m.%Y %H:%M:%S")]
    top = height - pad * 2 - line_height * len(footer)
    draw.rectangle([0, top, width, height], fill=(0, 0, 0, 170))
    for i, text in enumerate(footer):
        color = (255, 230, 0) if i == 0 else (230, 230, 230)
        draw.text((pad, top + pad + i * line_height), text, font=font, fill=color + (255,))
    return np.array(Image.alpha_composite(canvas, layer).convert("RGB"))


def _service_tags(ds: Dataset, config: ServiceConfig, processed_at: datetime) -> None:
    """Прил. 7."""
    ds.SpecificCharacterSet = UTF8
    ds.SeriesDescription = config.series_description
    ds.InstitutionName = config.name
    ds.InstitutionalDepartmentName = config.version
    ds.AcquisitionDate = processed_at.strftime("%Y%m%d")
    ds.AcquisitionTime = processed_at.strftime("%H%M%S")
    ds.OperatorsName = "AI"


def _copy_identifiers(target: Dataset, ref_ds: Dataset) -> None:
    for keyword in ("AccessionNumber", "PatientID", "IssuerOfPatientID", "FillerOrderNumberImagingServiceRequest"):
        if ref_ds.get(keyword):
            setattr(target, keyword, ref_ds.get(keyword))


def create_markup_sc(ref_ds: Dataset, rgb: np.ndarray, *, series_instance_uid: str, instance_number: int,
                     config: ServiceConfig, processed_at: datetime) -> Dataset:
    spacing = pixel_spacing(ref_ds)
    same_size = rgb.shape[:2] == (int(ref_ds.Rows), int(ref_ds.Columns))
    sex = ref_ds.get("PatientSex") or None
    laterality = ref_ds.get("Laterality") or ref_ds.get("ImageLaterality") or None
    orientation = ref_ds.get("PatientOrientation")
    orientation = tuple(orientation) if orientation and len(orientation) == 2 else ("L", "F")

    sc = hd.sc.SCImage(
        pixel_array=rgb.astype(np.uint8),
        photometric_interpretation=hd.PhotometricInterpretationValues.RGB,
        bits_allocated=8,
        coordinate_system=hd.CoordinateSystemNames.PATIENT,
        study_instance_uid=ref_ds.StudyInstanceUID,
        series_instance_uid=series_instance_uid,
        sop_instance_uid=hd.UID(),
        series_number=9001,
        instance_number=instance_number,
        manufacturer=config.name,
        patient_id=ref_ds.get("PatientID") or None,
        patient_name=ref_ds.get("PatientName") or None,
        patient_birth_date=ref_ds.get("PatientBirthDate") or None,
        patient_sex=sex if sex in ("M", "F", "O") else None,
        accession_number=ref_ds.get("AccessionNumber") or None,
        study_id=ref_ds.get("StudyID") or None,
        study_date=ref_ds.get("StudyDate") or None,
        study_time=ref_ds.get("StudyTime") or None,
        referring_physician_name=ref_ds.get("ReferringPhysicianName") or None,
        pixel_spacing=(spacing.row_mm, spacing.col_mm) if spacing and same_size else None,
        laterality=laterality if laterality in ("L", "R") else None,
        patient_orientation=orientation,
    )
    _service_tags(sc, config, processed_at)
    _copy_identifiers(sc, ref_ds)
    sc.Modality = config.additional_series_modality(str(ref_ds.get("Modality", "")))
    sc.BurnedInAnnotation = "YES"
    source = Dataset()
    source.ReferencedSOPClassUID = ref_ds.SOPClassUID
    source.ReferencedSOPInstanceUID = ref_ds.SOPInstanceUID
    sc.SourceImageSequence = [source]
    sc.DerivationDescription = "Автоматическая оценка качества DXA"
    return sc


# ---------------------------------------------------------------- SR


def _header(ds: Dataset) -> Dataset:
    header = Dataset()
    for keyword in ["PatientName", "PatientID", "PatientBirthDate", "PatientSex", "StudyInstanceUID", "StudyID",
                    "StudyDate", "StudyTime", "AccessionNumber", "ReferringPhysicianName", "SeriesInstanceUID",
                    "SOPClassUID", "SOPInstanceUID", "Modality"]:
        setattr(header, keyword, ds.get(keyword, "") or "")
    return header


def create_qc_sr(image_datasets: list[Dataset], study_result: QCResult, image_results: list[tuple[str, QCResult]],
                 *, series_instance_uid: str, config: ServiceConfig, processed_at: datetime) -> Dataset:
    values = {
        "modality": ", ".join(sorted({str(ds.get("Modality", "")) for ds in image_datasets})),
        "body-part": body_parts_text(image_datasets),
        "study-uid": str(image_datasets[0].StudyInstanceUID),
        "report-datetime": processed_at.strftime("%d.%m.%Y %H:%M:%S"),
        "warning-origin": config.conclusion_warning,
        "warning-usage": config.usage_warning,
        "service-name": config.name,
        "service-version": config.version,
        "service-purpose": config.purpose,
        "technical-data": technical_text(image_datasets),
        "technological-defects": defects_text(study_result),
        "description": description_text(image_results),
        "conclusion": conclusion_text(study_result),
        "user-guide": USER_GUIDE,
    }
    contains = hd.sr.RelationshipTypeValues.CONTAINS
    items = [hd.sr.TextContentItem(name=_code(code, meaning), value=values[code] or "—", relationship_type=contains)
             for code, meaning in SR_FIELDS]
    root = hd.sr.ContainerContentItem(name=REPORT_TITLE)
    root.ContentSequence = hd.sr.ContentSequence(items)

    sr = hd.sr.ComprehensiveSR(
        evidence=[_header(ds) for ds in image_datasets],
        content=root,
        series_instance_uid=series_instance_uid,
        series_number=9002,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer=config.name,
        is_complete=True,
        is_final=False,
        is_verified=False,
    )
    _service_tags(sr, config, processed_at)
    _copy_identifiers(sr, image_datasets[0])
    return sr


# ---------------------------------------------------------------- сообщения (прил. 4, 5)


def _iso(value: datetime | None) -> str | None:
    return value.astimezone().isoformat(timespec="seconds") if value else None


def report_notify_message(study_uid: str, series_uid: str, result: QCResult, config: ServiceConfig,
                          times: dict[str, datetime], report: str, conclusion: str) -> dict:
    """Прил. 4.1 (топик DICOMREPORTNOTIFY). Блок probParams для DXA — наше предложение, согласовать."""
    confidence = int(round(result.defect_probability * 100))
    return {
        "studyIUID": study_uid,
        "aiResult": {
            "seriesIUID": series_uid,
            "pathologyFlag": result.has_violations,
            "norma": 0 if result.has_violations else 1,
            "confidenceLevel": confidence,
            "modelId": config.model_id,
            "modelVersion": config.version,
            "report": report,
            "conclusion": conclusion,
            "dateTimeParams": {key: _iso(times.get(key)) for key in
                               ("downloadStartDT", "downloadEndDT", "processStartDT", "processEndDT")},
            "probParams": {
                "dxa_qc": {"dxa_qc_conf_level": confidence,
                           **{f"dxa_qc_{code}": int(code in result.defects) for code in DEFECTS}},
            },
        },
    }


def error_message(study_uid: str, category: str, description: str, config: ServiceConfig,
                  times: dict[str, datetime]) -> dict:
    """Прил. 5.1 (топик PUMCONSUMERERROR), категории ошибок — таблица 1 п. 3.5."""
    return {
        "studyIUID": study_uid,
        "aiResult": {
            "modelId": config.model_id,
            "error": category,
            "description": description,
            "dateTimeParams": {key: _iso(times.get(key)) for key in ("downloadStartDT", "downloadEndDT")},
        },
    }
