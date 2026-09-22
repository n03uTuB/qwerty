"""Разведка папки с DICOM: сводная таблица CSV и HTML-отчёт с превью и дампом тегов.

Цель — за 10 минут понять, что лежит в данных хакатона.
"""

from __future__ import annotations

import base64
import csv
import html
import io
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from pydicom.dataset import Dataset
from pydicom.uid import UID

from .markup import overlay_groups
from .qc import metadata_checks
from .reading import dicom_window, is_color, iter_dicom_files, pixel_spacing, read, to_display

KEY_TAGS = [
    "Modality", "Manufacturer", "ManufacturerModelName", "SeriesNumber", "SeriesDescription",
    "BodyPartExamined", "Laterality", "ImageLaterality", "ViewPosition", "PatientSex", "PatientAge",
    "PatientSize", "PatientWeight", "StudyDate",
]


def private_tags_report(ds: Dataset) -> str:
    """Приватные теги производителя. В бинарных значениях часто спрятан текст или XML — показываем превью."""
    lines = []
    for element in ds.iterall():
        if not element.tag.is_private:
            continue
        if element.tag.is_private_creator:
            lines.append(f"{element.tag} создатель блока: {element.value}")
            continue
        value = element.value
        if isinstance(value, (bytes, bytearray)):
            preview = bytes(value[:120])
            text = "".join(ch if 32 <= ord(ch) < 127 else "." for ch in preview.decode("latin-1"))
            lines.append(f"{element.tag} {element.VR} {len(value)} байт | hex {preview[:16].hex(' ')} | текст: {text}")
        else:
            lines.append(f"{element.tag} {element.VR} {element.name}: {value!r}"[:300])
    return "\n".join(lines)


def full_dump(ds: Dataset) -> str:
    meta = str(ds.file_meta) if getattr(ds, "file_meta", None) else "(нет file meta)"
    return f"{meta}\n\n{ds}"


def summarize_file(path: Path) -> tuple[dict, bytes | None, str]:
    ds = read(path)
    meta = getattr(ds, "file_meta", None)
    sop = ds.get("SOPClassUID") or (meta.get("MediaStorageSOPClassUID") if meta else None)
    ts = meta.get("TransferSyntaxUID") if meta else None

    row: dict = {"file": str(path)}
    row["sop_class"] = UID(sop).name if sop else ""
    row["transfer_syntax"] = UID(ts).name if ts else "?"
    row["compressed"] = bool(ts and UID(ts).is_compressed)
    for keyword in KEY_TAGS:
        row[keyword] = str(ds.get(keyword, "") or "")
    row["study_uid"] = str(ds.get("StudyInstanceUID", ""))
    row["series_uid"] = str(ds.get("SeriesInstanceUID", ""))

    flags = [f"{f.severity}: {f.title}" for f in metadata_checks(ds)]
    has_pixels = "PixelData" in ds
    row["has_pixels"] = has_pixels
    if has_pixels:
        row["size"] = f"{ds.get('Rows')}x{ds.get('Columns')}x{ds.get('NumberOfFrames', 1)}"
        row["photometric"] = str(ds.get("PhotometricInterpretation", ""))
        row["bits"] = f"{ds.get('BitsStored')}/{ds.get('BitsAllocated')} repr={ds.get('PixelRepresentation')}"
        spacing = pixel_spacing(ds)
        row["spacing_mm"] = f"{spacing.row_mm:g}x{spacing.col_mm:g} ({spacing.source})" if spacing else ""
        window = dicom_window(ds)
        row["window"] = f"{window[0]:g}/{window[1]:g}" if window else ""
        if "RescaleSlope" in ds or "RescaleIntercept" in ds:
            row["rescale"] = f"{ds.get('RescaleSlope', 1)}/{ds.get('RescaleIntercept', 0)}"
        if row["photometric"] == "MONOCHROME1":
            flags.append("info: MONOCHROME1 — шкала инвертирована")
        if is_color(ds):
            flags.append("info: цветное изображение — вероятно, скриншот с разметкой")
    row["burned_in"] = str(ds.get("BurnedInAnnotation", ""))
    row["overlays"] = ",".join(f"{g:04X}" for g in overlay_groups(ds))
    private = [el for el in ds.iterall() if el.tag.is_private]
    row["private_tags"] = len(private)
    row["private_creators"] = "; ".join(sorted({str(el.value) for el in private if el.tag.is_private_creator}))
    row["encapsulated_doc"] = str(ds.get("MIMETypeOfEncapsulatedDocument", ""))
    row["is_sr"] = ds.get("Modality") == "SR" or "ContentSequence" in ds

    thumbnail = None
    if has_pixels:
        try:
            image = Image.fromarray(to_display(ds))
            image.thumbnail((256, 256))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            thumbnail = buffer.getvalue()
        except Exception as exc:  # чаще всего — нет декодера для сжатия
            flags.append(f"error: не удалось декодировать пиксели: {exc}")
    row["flags"] = " | ".join(flags)

    dump = full_dump(ds)
    private_report = private_tags_report(ds)
    if private_report:
        dump += "\n\n=== Приватные теги ===\n" + private_report
    return row, thumbnail, dump


_TABLE_COLUMNS = [
    "sop_class", "Modality", "Manufacturer", "SeriesDescription", "BodyPartExamined", "Laterality",
    "size", "photometric", "bits", "transfer_syntax", "spacing_mm", "window", "overlays",
    "private_tags", "encapsulated_doc", "flags",
]

_CSS = """
body{font-family:system-ui,sans-serif;margin:16px;color:#1b1b1b;background:#fafafa}
table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccc;padding:4px;vertical-align:top}
th{background:#eee;position:sticky;top:0}img{max-width:256px}pre{font-size:11px;max-height:500px;overflow:auto}
.flags{color:#a33;white-space:pre-line}.wrap{overflow-x:auto}
"""


def build_report(root: str | Path, out_dir: str | Path) -> list[dict]:
    root, out = Path(root), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, cards = [], []
    for path in iter_dicom_files(root):
        try:
            row, thumb, dump = summarize_file(path)
        except Exception as exc:
            row, thumb, dump = {"file": str(path), "flags": f"error: не прочитан: {exc}"}, None, ""
        row["file"] = str(path.relative_to(root)) if root.is_dir() else path.name
        rows.append(row)
        cards.append((row, thumb, dump))

    fieldnames = sorted({key for row in rows for key in row}, key=lambda k: (k != "file", k))
    with open(out / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    studies = defaultdict(lambda: defaultdict(list))
    for row in rows:
        studies[row.get("study_uid", "")][(row.get("SeriesNumber", ""), row.get("SeriesDescription", ""), row.get("Modality", ""))].append(row["file"])

    parts = [f"<title>DICOM разведка</title><style>{_CSS}</style><h1>Разведка DICOM: {html.escape(str(root))}</h1>"]
    parts.append(f"<p>Файлов DICOM: {len(rows)}. Таблица: summary.csv</p><h2>Исследования и серии</h2><ul>")
    for study, series in studies.items():
        parts.append(f"<li>Study <code>{html.escape(study)}</code><ul>")
        for (number, description, modality), files in sorted(series.items(), key=lambda kv: str(kv[0][0])):
            parts.append(f"<li>#{html.escape(str(number))} {html.escape(description)} [{html.escape(modality)}] — файлов: {len(files)}</li>")
        parts.append("</ul></li>")
    parts.append("</ul><h2>Файлы</h2><div class='wrap'><table><tr><th>превью</th><th>файл</th>")
    parts.extend(f"<th>{c}</th>" for c in _TABLE_COLUMNS)
    parts.append("</tr>")
    for row, thumb, dump in cards:
        img = f"<img src='data:image/png;base64,{base64.b64encode(thumb).decode()}'>" if thumb else ""
        parts.append(f"<tr><td>{img}</td><td>{html.escape(row['file'])}<details><summary>теги</summary><pre>{html.escape(dump)}</pre></details></td>")
        for column in _TABLE_COLUMNS:
            value = html.escape(str(row.get(column, "")))
            if column == "flags":
                value = f"<span class='flags'>{value.replace(' | ', chr(10))}</span>"
            parts.append(f"<td>{value}</td>")
        parts.append("</tr>")
    parts.append("</table></div>")
    (out / "report.html").write_text("".join(parts), encoding="utf-8")
    return rows


def print_overview(rows: list[dict]) -> None:
    def show(title, counter):
        print(f"\n{title}:")
        for value, count in counter.most_common():
            print(f"  {count:5d}  {value or '(пусто)'}")

    print(f"Файлов DICOM: {len(rows)}")
    for key, title in [("sop_class", "Типы объектов (SOP Class)"), ("Modality", "Модальность"),
                       ("Manufacturer", "Производитель"), ("photometric", "Photometric Interpretation"),
                       ("transfer_syntax", "Transfer Syntax"), ("BodyPartExamined", "Область")]:
        show(title, Counter(str(r.get(key, "")) for r in rows))
    show("Флаги", Counter(flag for r in rows for flag in str(r.get("flags", "")).split(" | ") if flag))
