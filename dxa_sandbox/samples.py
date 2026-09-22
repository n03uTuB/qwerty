"""Синтетическое «DXA-подобное» исследование с типичными ловушками формата.

Это НЕ медицинские данные: фантом из прямоугольников и эллипсов, чтобы отладить код до хакатона.
Все файлы относятся к одному исследованию (StudyInstanceUID) и разным сериям.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import JPEG2000Lossless, RLELossless, ExplicitVRLittleEndian, generate_uid
from scipy import ndimage

from .geometry import image_center, rotate_image, rotate_points

DX_FOR_PRESENTATION = "1.2.840.10008.5.1.4.1.1.1.1"
SECONDARY_CAPTURE = "1.2.840.10008.5.1.4.1.1.7"
ENCAPSULATED_PDF = "1.2.840.10008.5.1.4.1.1.104.1"

VERTEBRAE = ["T12", "L1", "L2", "L3", "L4", "L5"]


# ---------------------------------------------------------------- фантомы


def _shape_mask(shape: tuple[int, int], draw_fn) -> np.ndarray:
    rows, cols = shape
    image = Image.new("L", (cols, rows), 0)
    draw_fn(ImageDraw.Draw(image))
    return np.array(image) > 0


def spine_phantom(rows: int = 400, cols: int = 300, tilt_deg: float = 0.0, seed: int = 0):
    """Поясничный отдел в прямой проекции. Возвращает 12-битное изображение и полигоны позвонков."""
    rng = np.random.default_rng(seed)
    shape = (rows, cols)
    img = np.full(shape, 500.0)
    img += _shape_mask(shape, lambda d: d.rectangle([cols * 0.05, 0, cols * 0.95, rows], fill=1)) * 500

    for side in (-1, 1):  # гребни подвздошных костей
        cx, cy = cols / 2 + side * cols * 0.36, rows * 1.02
        img += _shape_mask(
            shape,
            lambda d, cx=cx, cy=cy: d.ellipse([cx - cols * 0.2, cy - rows * 0.22, cx + cols * 0.2, cy + rows * 0.22], fill=1),
        ) * 900

    height, gap, top = rows * 0.12, rows * 0.03, rows * 0.04
    t = np.tan(np.radians(tilt_deg))
    polygons = {}
    for i, name in enumerate(VERTEBRAE):
        y0 = top + i * (height + gap)
        y1 = y0 + height
        yc = (y0 + y1) / 2
        w = cols * 0.2 * (1 + 0.06 * i)
        cx = cols / 2 + t * (yc - rows / 2)
        dx0, dx1 = t * (y0 - yc), t * (y1 - yc)
        poly = [(cx - w / 2 + dx0, y0), (cx + w / 2 + dx0, y0), (cx + w / 2 + dx1, y1), (cx - w / 2 + dx1, y1)]
        img += _shape_mask(shape, lambda d, poly=poly: d.polygon(poly, fill=1)) * 1500
        polygons[name] = poly

    y_rib = top + height * 0.4  # рёбра у T12
    for side in (-1, 1):
        x_start = cols / 2 + side * cols * 0.11 + t * (y_rib - rows / 2)
        pts = [(x_start, y_rib), (x_start + side * cols * 0.3, y_rib + rows * 0.08)]
        img += _shape_mask(shape, lambda d, pts=pts: d.line(pts, fill=1, width=max(3, rows // 60))) * 500

    img += rng.normal(0, 40, shape)
    return np.clip(img, 0, 4095).astype(np.uint16), polygons


def hip_phantom(rows: int = 360, cols: int = 300, side: str = "L", external_rotation_deg: float = 0.0, seed: int = 1):
    """Проксимальный отдел бедра. Чем больше наружная ротация, тем крупнее малый вертел."""
    rng = np.random.default_rng(seed)
    shape = (rows, cols)
    c, r = cols, rows
    base = 1000.0
    img = np.full(shape, base)

    def add(draw_fn, value):
        # кости не суммируются, а перекрываются — иначе на стыках появляются засвеченные пятна
        nonlocal img
        img = np.maximum(img, base + _shape_mask(shape, draw_fn) * value)

    add(lambda d: d.ellipse([c * 0.0, -r * 0.25, c * 0.5, r * 0.2], fill=1), 700)  # таз
    add(lambda d: d.ellipse([c * 0.19, r * 0.15, c * 0.41, r * 0.41], fill=1), 1400)  # головка
    add(lambda d: d.polygon([(c * 0.33, r * 0.20), (c * 0.58, r * 0.30), (c * 0.60, r * 0.46), (c * 0.36, r * 0.36)], fill=1), 1200)
    add(lambda d: d.ellipse([c * 0.52, r * 0.26, c * 0.74, r * 0.50], fill=1), 1300)  # большой вертел
    add(lambda d: d.polygon([(c * 0.50, r * 0.40), (c * 0.68, r * 0.40), (c * 0.66, r * 1.0), (c * 0.50, r * 1.0)], fill=1), 1500)
    k = 1 + abs(external_rotation_deg) / 10
    add(lambda d: d.ellipse([c * 0.50 - c * 0.05 * k, r * 0.52, c * 0.52, r * 0.52 + r * 0.06 * k], fill=1), 900)
    img = ndimage.gaussian_filter(img, sigma=2) + rng.normal(0, 40, shape)

    # ROI шейки перпендикулярно её оси и общая ROI бедра
    p0, p1 = np.array([c * 0.30, r * 0.28]), np.array([c * 0.62, r * 0.40])
    length = np.linalg.norm(p1 - p0)
    axis = (p1 - p0) / length
    normal = np.array([-axis[1], axis[0]])
    center = p0 + axis * length * 0.45
    neck = [tuple(center + sa * axis * c * 0.03 + sn * normal * c * 0.12) for sa, sn in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    total = [(c * 0.15, r * 0.12), (c * 0.80, r * 0.12), (c * 0.80, r * 0.70), (c * 0.15, r * 0.70)]

    stored = np.clip(img, 0, 4095).astype(np.uint16)
    if side == "R":  # правое бедро — зеркально
        stored = stored[:, ::-1].copy()
        neck = [(cols - 1 - x, y) for x, y in neck]
        total = [(cols - 1 - x, y) for x, y in total]
    return stored, {"neck_roi": [tuple(map(float, p)) for p in neck], "total_hip": total}


# ---------------------------------------------------------------- DICOM


def _new_dataset(sop_class_uid: str, study_uid: str, *, series_number: int, modality: str,
                 series_description: str, **attributes) -> Dataset:
    now = datetime.now()
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.SpecificCharacterSet = "ISO_IR 192"  # UTF-8
    ds.SOPClassUID = sop_class_uid
    ds.SOPInstanceUID = generate_uid()
    ds.PatientName = "ANON^PHANTOM"
    ds.PatientID = "ANON0001"
    ds.PatientBirthDate = ""
    ds.StudyInstanceUID = study_uid
    ds.StudyDate = ds.ContentDate = now.strftime("%Y%m%d")
    ds.StudyTime = ds.ContentTime = now.strftime("%H%M%S")
    ds.StudyID = "1"
    ds.AccessionNumber = "ACC0001"
    ds.ReferringPhysicianName = ""
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = series_number
    ds.InstanceNumber = 1
    ds.Modality = modality
    ds.SeriesDescription = series_description
    ds.Manufacturer = "SandboxDXA"
    ds.ManufacturerModelName = "Phantom-1"
    for keyword, value in attributes.items():
        setattr(ds, keyword, value)
    return ds


def _save(ds: Dataset, path: Path) -> Path:
    ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(path, enforce_file_format=True)
    return path


def _minimal_pdf(lines: list[str]) -> bytes:
    content = "BT /F1 12 Tf 50 780 Td 16 TL " + " ".join(f"({t}) Tj T*" for t in lines) + " ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(content)} >>\nstream\n{content}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{offset:010d} 00000 n \n" for offset in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


SAMPLE_NOTES = {
    "01_spine_raw.dcm": "DX For Presentation, MONOCHROME2, 12 бит в 16, окно в тегах, "
                        "ImagerPixelSpacing, приватный блок производителя с BMD и XML",
    "02_hip_left_mono1_compressed.dcm": "MONOCHROME1 (инверсия), сжатие JPEG 2000 или RLE, "
                                        "PixelSpacing, НЕТ пола пациента",
    "03_hip_right_overlay.dcm": "8 бит, ROI шейки и бедра в overlay 6000, сильная наружная ротация "
                                "(крупный малый вертел), НЕТ возраста",
    "04_spine_report_sc_rgb.dcm": "Secondary Capture RGB: скриншот с цветной разметкой, наклон 6°, "
                                  "НЕТ размера пикселя",
    "05_report_pdf.dcm": "Encapsulated PDF с таблицей результатов",
    "notes.txt": "не DICOM — сканер должен его пропустить",
}


def make_samples(out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    study = generate_uid()
    paths = {}
    orientation = ["L", "F"]  # для AP: столбцы — к левому боку пациента, строки — к ногам

    # 1. сырой снимок позвоночника
    spine, vertebrae = spine_phantom()
    ds = _new_dataset(DX_FOR_PRESENTATION, study, series_number=1, modality="BMD",
                      series_description="AP Spine L1-L4",
                      PatientSex="F", PatientAge="067Y", PatientSize="1.62", PatientWeight="58",
                      BodyPartExamined="LSPINE", ViewPosition="AP", PatientOrientation=orientation,
                      ImagerPixelSpacing=["0.5", "0.5"], WindowCenter="1800", WindowWidth="3000",
                      PresentationLUTShape="IDENTITY", BurnedInAnnotation="NO")
    ds.set_pixel_data(spine, photometric_interpretation="MONOCHROME2", bits_stored=12)
    block = ds.private_block(0x0019, "SANDBOX_DXA_ANALYSIS", create=True)
    block.add_new(0x01, "LO", "L1-L4")
    block.add_new(0x02, "DS", ["0.912", "0.955", "1.001", "1.043"])
    rois = "".join(
        f'<roi name="{name}" y0="{poly[0][1]:.1f}" y1="{poly[2][1]:.1f}"/>' for name, poly in vertebrae.items()
    )
    xml = f"<analysis>{rois}</analysis>".encode()
    block.add_new(0x10, "OB", xml + b" " * (len(xml) % 2))
    paths["01_spine_raw.dcm"] = _save(ds, out / "01_spine_raw.dcm")

    # 2. левое бедро: MONOCHROME1 + сжатие, без пола
    hip_left, _ = hip_phantom(side="L")
    ds = _new_dataset(DX_FOR_PRESENTATION, study, series_number=2, modality="BMD",
                      series_description="Left Femur", PatientAge="067Y",
                      BodyPartExamined="HIP", Laterality="L", PatientOrientation=orientation,
                      PixelSpacing=["0.5", "0.5"])
    ds.set_pixel_data((4095 - hip_left).astype(np.uint16), photometric_interpretation="MONOCHROME1", bits_stored=12)
    try:
        ds.compress(JPEG2000Lossless)
    except Exception:  # нет кодировщика JPEG 2000 — RLE есть всегда
        ds.compress(RLELossless)
    paths["02_hip_left_mono1_compressed.dcm"] = _save(ds, out / "02_hip_left_mono1_compressed.dcm")

    # 3. правое бедро: разметка в overlay
    hip_right, rois_right = hip_phantom(side="R", external_rotation_deg=20, seed=2)
    rows, cols = hip_right.shape
    ds = _new_dataset(DX_FOR_PRESENTATION, study, series_number=3, modality="BMD",
                      series_description="Right Femur", PatientSex="F",
                      BodyPartExamined="HIP", Laterality="R", PatientOrientation=orientation,
                      ImagerPixelSpacing=["0.5", "0.5"])
    ds.set_pixel_data((hip_right / 4095 * 255).astype(np.uint8), photometric_interpretation="MONOCHROME2", bits_stored=8)
    mask = _shape_mask(
        (rows, cols),
        lambda d: (
            d.line(rois_right["neck_roi"] + [rois_right["neck_roi"][0]], fill=1, width=2),
            d.line(rois_right["total_hip"] + [rois_right["total_hip"][0]], fill=1, width=2),
        ),
    )
    packed = np.packbits(mask.ravel().astype(np.uint8), bitorder="little").tobytes()
    ds.add_new((0x6000, 0x0010), "US", rows)
    ds.add_new((0x6000, 0x0011), "US", cols)
    ds.add_new((0x6000, 0x0022), "LO", "QC ROI")
    ds.add_new((0x6000, 0x0040), "CS", "G")
    ds.add_new((0x6000, 0x0050), "SS", [1, 1])
    ds.add_new((0x6000, 0x0100), "US", 1)
    ds.add_new((0x6000, 0x0102), "US", 0)
    ds.add_new((0x6000, 0x3000), "OW", packed + b"\x00" * (len(packed) % 2))
    paths["03_hip_right_overlay.dcm"] = _save(ds, out / "03_hip_right_overlay.dcm")

    # 4. скриншот отчёта с цветной разметкой: пациент лежит с наклоном 6°
    spine_straight, polys = spine_phantom(seed=3)
    spine_tilted = rotate_image(spine_straight.astype(np.float32), 6.0)
    center = image_center(spine_tilted.shape)
    polys = {name: rotate_points(poly, 6.0, center) for name, poly in polys.items()}
    gray = np.clip((spine_tilted.astype(np.float64) - 300) / 3000 * 255, 0, 255).astype(np.uint8)
    picture = Image.fromarray(np.stack([gray] * 3, axis=-1))
    draw = ImageDraw.Draw(picture)
    for upper, lower in zip(VERTEBRAE[:-1], VERTEBRAE[1:]):
        y = (polys[upper][2][1] + polys[lower][0][1]) / 2
        xc = (polys[upper][2][0] + polys[lower][0][0] + polys[upper][3][0] + polys[lower][1][0]) / 4
        draw.line([(xc - 60, y), (xc + 60, y)], fill=(0, 230, 0), width=2)
    for name in ("L1", "L2", "L3", "L4"):
        poly = polys[name]
        draw.text((poly[1][0] + 25, (poly[0][1] + poly[2][1]) / 2 - 6), name, fill=(255, 230, 0))
    xs = [x for name in ("L1", "L2", "L3", "L4") for x, _ in polys[name]]
    draw.rectangle([min(xs) - 8, polys["L1"][0][1] - 4, max(xs) + 8, polys["L4"][2][1] + 4], outline=(230, 0, 0), width=2)
    ds = _new_dataset(SECONDARY_CAPTURE, study, series_number=4, modality="OT",
                      series_description="Spine report screenshot",
                      PatientSex="F", PatientAge="067Y", BodyPartExamined="LSPINE",
                      BurnedInAnnotation="YES", ConversionType="WSD")
    ds.set_pixel_data(np.array(picture), photometric_interpretation="RGB", bits_stored=8)
    paths["04_spine_report_sc_rgb.dcm"] = _save(ds, out / "04_spine_report_sc_rgb.dcm")

    # 5. PDF-отчёт аппарата
    pdf = _minimal_pdf(["DXA report (synthetic phantom)", "L1-L4 BMD 0.978 g/cm2  T-score -1.6",
                        "Left femoral neck BMD 0.701 g/cm2  T-score -1.9"])
    ds = _new_dataset(ENCAPSULATED_PDF, study, series_number=5, modality="DOC",
                      series_description="DXA results report", PatientSex="F", PatientAge="067Y",
                      BurnedInAnnotation="YES", MIMETypeOfEncapsulatedDocument="application/pdf",
                      DocumentTitle="DXA report")
    ds.ConceptNameCodeSequence = []
    ds.EncapsulatedDocumentLength = len(pdf)
    ds.EncapsulatedDocument = pdf + b"\x00" * (len(pdf) % 2)
    paths["05_report_pdf.dcm"] = _save(ds, out / "05_report_pdf.dcm")

    (out / "notes.txt").write_text("не DICOM", encoding="utf-8")
    paths["notes.txt"] = out / "notes.txt"
    (out / "SAMPLES.md").write_text(
        "# Синтетические примеры (фантом, не медицинские данные)\n\n"
        + "\n".join(f"- `{name}` — {note}" for name, note in SAMPLE_NOTES.items())
        + "\n",
        encoding="utf-8",
    )
    return paths
