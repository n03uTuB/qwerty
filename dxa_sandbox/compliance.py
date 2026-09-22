"""Самопроверка результата по п. 3.3, 3.4 и прил. 7 «Базовых функциональных требований» ЦДТ."""

from __future__ import annotations

import re
from pathlib import Path

import pydicom
from pydicom.dataset import Dataset

from .config import MAX_UID_LENGTH, ServiceConfig
from .writers import SR_FIELDS

SECONDARY_CAPTURE = "1.2.840.10008.5.1.4.1.1.7"


def _text(ds: Dataset, keyword: str) -> str:
    return str(ds.get(keyword, "") or "")


def _service_tag_problems(ds: Dataset, config: ServiceConfig, where: str) -> list[str]:
    problems = []
    expected = {
        "SeriesDescription": config.series_description,
        "InstitutionName": config.name,
        "InstitutionalDepartmentName": config.version,
        "OperatorsName": "AI",
    }
    for keyword, value in expected.items():
        if _text(ds, keyword) != value:
            problems.append(f"{where}: {keyword} = «{_text(ds, keyword)}», ожидалось «{value}»")
    if not re.fullmatch(r"\d{8}", _text(ds, "AcquisitionDate")):
        problems.append(f"{where}: AcquisitionDate не в формате YYYYMMDD")
    if not re.fullmatch(r"\d{6}(\.\d+)?", _text(ds, "AcquisitionTime")):
        problems.append(f"{where}: AcquisitionTime не в формате HHMMSS")
    for keyword in ("SeriesInstanceUID", "SOPInstanceUID"):
        if len(_text(ds, keyword)) > MAX_UID_LENGTH:
            problems.append(f"{where}: {keyword} длиннее {MAX_UID_LENGTH} символов")
    return problems


def check_outputs(originals: list[tuple[Path, Dataset]], sc_paths: list[Path], sr_path: Path,
                  config: ServiceConfig) -> dict:
    problems, notes = [], []
    by_instance = {str(ds.SOPInstanceUID): (path, ds) for path, ds in originals}
    scs = [pydicom.dcmread(p) for p in sc_paths]

    if len({str(ds.SeriesInstanceUID) for ds in scs}) != 1:
        problems.append("Дополнительная серия должна быть одна (п. 3.4)")
    for path, sc in zip(sc_paths, scs):
        where = path.name
        problems += _service_tag_problems(sc, config, where)
        if str(sc.SOPClassUID) != SECONDARY_CAPTURE:
            problems.append(f"{where}: SOPClassUID не Secondary Capture")
        if not str(sc.SeriesInstanceUID).endswith(f".{config.model_id}.1.1"):
            problems.append(f"{where}: SeriesInstanceUID не по маске {{OriginalSeriesUID}}.{{modelId}}.{{addId}}.{{count}}")
        source = sc.get("SourceImageSequence")
        original = by_instance.get(str(source[0].ReferencedSOPInstanceUID)) if source else None
        if original is None:
            problems.append(f"{where}: нет ссылки на исходное изображение")
            continue
        _, ref = original
        if _text(sc, "Modality") != config.additional_series_modality(_text(ref, "Modality")):
            problems.append(f"{where}: Modality не соответствует прил. 7")
        for keyword in ("StudyInstanceUID", "PatientID", "AccessionNumber"):
            if _text(sc, keyword) != _text(ref, keyword):
                problems.append(f"{where}: {keyword} не совпадает с оригиналом")
        if (sc.Rows, sc.Columns) != (ref.Rows, ref.Columns):
            problems.append(f"{where}: разрешение отличается от исходного")

    original_bytes = sum(path.stat().st_size for path, _ in originals)
    sc_bytes = sum(p.stat().st_size for p in sc_paths)
    if sc_bytes > original_bytes:
        if all(int(ds.Rows) < 512 or int(ds.Columns) < 512 for _, ds in originals):
            notes.append(f"Доп. серия больше оригинала ({sc_bytes} > {original_bytes} байт): "
                         "допустимо для изображений меньше 512×512 (п. 3.4)")
        else:
            problems.append(f"Доп. серия больше оригинала ({sc_bytes} > {original_bytes} байт), п. 3.4")

    sr = pydicom.dcmread(sr_path)
    problems += _service_tag_problems(sr, config, sr_path.name)
    if _text(sr, "Modality") != "SR":
        problems.append("SR: Modality должна быть SR")
    meanings = [str(item.ConceptNameCodeSequence[0].CodeMeaning) for item in sr.get("ContentSequence", [])]
    if meanings != [meaning for _, meaning in SR_FIELDS]:
        problems.append("SR: состав или порядок полей не соответствует п. 3.3")
    texts = " ".join(str(item.get("TextValue", "")) for item in sr.get("ContentSequence", []))
    for warning in (config.conclusion_warning, config.usage_warning):
        if warning not in texts:
            problems.append(f"SR: нет предупреждения «{warning}»")
    return {"problems": problems, "notes": notes}
