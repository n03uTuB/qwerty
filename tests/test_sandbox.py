import json

import numpy as np
import pydicom
import pytest

from dxa_sandbox import geometry, inspection, markup, pipeline, qc, reading, samples
from dxa_sandbox.config import series_uid_by_mask


@pytest.fixture(scope="module")
def sample_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("samples")
    samples.make_samples(directory)
    return directory


def _read(sample_dir, name):
    return reading.read(sample_dir / name)


def test_scanner_finds_only_dicom(sample_dir):
    files = [p.name for p in reading.iter_dicom_files(sample_dir)]
    assert len(files) == 5
    assert "notes.txt" not in files


def test_all_images_decode(sample_dir):
    for path in reading.iter_dicom_files(sample_dir):
        ds = reading.read(path)
        if "PixelData" in ds:
            image = reading.to_display(ds)
            assert image.dtype == np.uint8 and image.max() > image.min()


def test_compressed_monochrome1_bone_is_bright(sample_dir):
    ds = _read(sample_dir, "02_hip_left_mono1_compressed.dcm")
    assert ds.file_meta.TransferSyntaxUID.is_compressed
    assert ds.PhotometricInterpretation == "MONOCHROME1"
    image = reading.to_display(ds)
    rows, cols = image.shape
    shaft, soft_tissue = image[int(rows * 0.8), int(cols * 0.58)], image[int(rows * 0.8), int(cols * 0.1)]
    assert shaft > soft_tissue + 50


def test_overlay_mask(sample_dir):
    ds = _read(sample_dir, "03_hip_right_overlay.dcm")
    assert markup.overlay_groups(ds) == [0x6000]
    mask = markup.overlay_mask(ds)
    assert mask.shape == (ds.Rows, ds.Columns) and mask.sum() > 100
    assert len(markup.components(mask)) >= 1


def test_color_markup_lines(sample_dir):
    ds = _read(sample_dir, "04_spine_report_sc_rgb.dcm")
    masks = markup.color_markup_masks(reading.to_display(ds))
    assert {"green", "red", "yellow"} <= set(masks)
    assert len(markup.horizontal_line_rows(masks["green"])) == 5  # границы T12/L1 ... L4/L5


def test_spine_tilt_demo(sample_dir):
    straight = reading.to_display(_read(sample_dir, "01_spine_raw.dcm"))
    angle, _ = geometry.spine_axis_tilt(straight)
    assert abs(angle) < 2

    sc = reading.to_display(_read(sample_dir, "04_spine_report_sc_rgb.dcm"))
    clean = markup.remove_markup(sc, np.logical_or.reduce(list(markup.color_markup_masks(sc).values())))
    angle, _ = geometry.spine_axis_tilt(clean)
    assert abs(angle - 6) < 2


def test_metadata_checks(sample_dir):
    codes = {f.code for f in qc.metadata_checks(_read(sample_dir, "02_hip_left_mono1_compressed.dcm"))}
    assert "missing_patient_sex" in codes
    codes = {f.code for f in qc.metadata_checks(_read(sample_dir, "04_spine_report_sc_rgb.dcm"))}
    assert "no_pixel_spacing" in codes


def test_private_tags_and_pdf(sample_dir, tmp_path):
    report = inspection.private_tags_report(_read(sample_dir, "01_spine_raw.dcm"))
    assert "SANDBOX_DXA_ANALYSIS" in report and "<analysis>" in report
    pdf = reading.extract_encapsulated_document(_read(sample_dir, "05_report_pdf.dcm"), tmp_path / "r.pdf")
    assert pdf.read_bytes().startswith(b"%PDF") and pdf.read_bytes().rstrip().endswith(b"%%EOF")


def test_inspection_report(sample_dir, tmp_path):
    rows = inspection.build_report(sample_dir, tmp_path)
    assert len(rows) == 5
    assert (tmp_path / "report.html").stat().st_size > 1000
    assert (tmp_path / "summary.csv").exists()


def test_qc_pipeline_outputs(sample_dir, tmp_path):
    outputs = pipeline.process_path(sample_dir, tmp_path)
    assert len(outputs) == 1
    (files,) = outputs.values()
    spine = _read(sample_dir, "01_spine_raw.dcm")

    scs = [pydicom.dcmread(f) for f in files if f.name.endswith("_sc.dcm")]
    assert len(scs) == 4
    assert {ds.StudyInstanceUID for ds in scs} == {spine.StudyInstanceUID}
    assert len({ds.SeriesInstanceUID for ds in scs}) == 1  # одна дополнительная серия
    # маска прил. 7: исходный UID обрезается до 56 символов, итог ≤ 64
    assert scs[0].SeriesInstanceUID == series_uid_by_mask(str(spine.SeriesInstanceUID), 1000)
    assert scs[0].SeriesInstanceUID.startswith(str(spine.SeriesInstanceUID)[:50])
    assert len(scs[0].SeriesInstanceUID) <= 64
    assert scs[0].OperatorsName == "AI" and scs[0].InstitutionName == "DXA-QC"

    sr = pydicom.dcmread(next(f for f in files if f.name.endswith("_sr.dcm")))
    assert sr.Modality == "SR" and sr.StudyInstanceUID == spine.StudyInstanceUID
    texts = " ".join(str(el.value) for el in sr.iterall() if el.keyword in ("TextValue", "CodeMeaning"))
    assert "Технологические дефекты при выполнении исследования" in texts  # кириллица пережила UTF-8
    assert "Нарушение укладки и позиционирования" in texts
    assert "В исследовательских целях" in texts

    report = json.loads((files[0].parent / "qc_report.json").read_text(encoding="utf-8"))
    assert report["has_violations"] is True
    assert "positioning" in report["defects"]
    assert report["compliance"]["problems"] == []

    notify = json.loads((files[0].parent / "dicom_report_notify.json").read_text(encoding="utf-8"))
    assert notify["aiResult"]["pathologyFlag"] is True and notify["aiResult"]["modelId"] == 1000
    assert notify["aiResult"]["seriesIUID"] == str(scs[0].SeriesInstanceUID)


def test_pum_integration_uses_asmt_modality(sample_dir, tmp_path):
    from dataclasses import replace

    from dxa_sandbox.config import DEFAULT

    (files,) = pipeline.process_path(sample_dir, tmp_path, replace(DEFAULT, integration="pum")).values()
    assert {pydicom.dcmread(f).Modality for f in files if f.name.endswith("_sc.dcm")} == {"ASMT"}
    report = json.loads((files[0].parent / "qc_report.json").read_text(encoding="utf-8"))
    assert report["compliance"]["problems"] == []


def test_study_without_images_returns_error_message(sample_dir, tmp_path):
    assert pipeline.process_study([sample_dir / "05_report_pdf.dcm"], tmp_path) == []
    message = json.loads((tmp_path / "pum_consumer_error.json").read_text(encoding="utf-8"))
    assert message["aiResult"]["error"] == "Series error"
