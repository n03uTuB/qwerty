# Шпаргалка по DICOM для задачи DXA

## Структура

```
Patient ─ Study (StudyInstanceUID) ─ Series (SeriesInstanceUID) ─ Instance = файл (SOPInstanceUID)
```

- Одно **исследование** DXA обычно состоит из нескольких серий: позвоночник, левое и правое бедро, отчёт.
- **SOPClassUID** — тип объекта:

| UID | Что это |
|---|---|
| 1.2.840.10008.5.1.4.1.1.1.1 | Digital X-Ray For Presentation |
| 1.2.840.10008.5.1.4.1.1.1 | Computed Radiography |
| 1.2.840.10008.5.1.4.1.1.7 | Secondary Capture — «скриншот», часто с вшитой разметкой |
| 1.2.840.10008.5.1.4.1.1.88.33 | Comprehensive SR — структурированный отчёт |
| 1.2.840.10008.5.1.4.1.1.104.1 | Encapsulated PDF |

- **Модальность** DXA по стандарту — `BMD`, но на практике встречаются `OT`, `DX`, `CR`.

## Пиксели: порядок преобразований

```
stored → Modality LUT (RescaleSlope/Intercept) → VOI LUT (окно) → Presentation (MONOCHROME1 = инверсия) → экран
```

| Тег | Ловушка |
|---|---|
| TransferSyntaxUID (в file meta) | сжатие: JPEG / JPEG 2000 / RLE — нужны `pylibjpeg-*` |
| PhotometricInterpretation | MONOCHROME1 — кость тёмная; RGB / YBR_FULL_422 — цветной снимок |
| BitsStored / BitsAllocated / PixelRepresentation | 12 бит в 16; знаковые значения |
| WindowCenter / WindowWidth | могут быть многозначными — берите первое значение |
| PixelSpacing / ImagerPixelSpacing | порядок: **[строки, столбцы]** = [по вертикали, по горизонтали], в мм |
| NumberOfFrames | pixel_array становится 3D |
| PatientOrientation | направления осей изображения, например `L\F` |
| BurnedInAnnotation | YES — текст или разметка вшиты в пиксели |

## Где искать разметку аппарата

1. **Overlay** — группы `6000–601E`: `(60xx,0010)` Rows, `(60xx,0011)` Columns, `(60xx,0050)` Origin (с 1), `(60xx,3000)` данные (биты от младшего к старшему).
2. **Цветные линии в пикселях** — маска по насыщенности и оттенку (`markup.color_markup_masks`).
3. **SR** — дерево `ContentSequence`: ValueType CONTAINER / TEXT / NUM / CODE / SCOORD (координаты).
4. **PDF** — `EncapsulatedDocument`.
5. **Приватные теги** — нечётные группы, блок начинается с тега-создателя `(gggg,00xx)`. Бинарные `OB/UN` часто содержат текст или XML: смотрите `dump --private`.

## Приватные теги

- Номер элемента зависит от блока: `(0019,1002)` в одном файле и `(0019,1102)` в другом могут быть одним и тем же полем. Ищите по **имени создателя**, а не по номеру: `ds.private_block(0x0019, "CREATOR")`.
- Неизвестный VR при Implicit VR Little Endian читается как `UN` (байты).

## Свой результат

- Новые UID: `pydicom.uid.generate_uid()`.
- **Оставить** StudyInstanceUID и данные пациента исходника, **новый** SeriesInstanceUID для своей серии.
- Все картинки одного прогона кладите в **одну** серию, отчёт — отдельной серией.
- `SpecificCharacterSet = "ISO_IR 192"` (UTF-8), иначе кириллица сломается.
- Ссылка на исходник: `SourceImageSequence` (в SC) или evidence (в SR).
- Коды в SR: для своих понятий используйте локальную схему (`99XXX`), общие берите из DCM, LOINC, SNOMED.
- Проверка валидности: `dciodvfy` (dicom3tools) или открыть файл в Weasis / Orthanc.

## Обезличенные данные

- Нет имени, даты рождения, иногда пола и возраста. PatientAge хранится в формате `067Y`.
- Даты могут быть сдвинуты. UID могут быть переписаны, но связь Study → Series должна сохраниться.
- Код не должен падать на пустых тегах: `ds.get("PatientSex") or None`.

## Сеть

| Способ | Когда |
|---|---|
| C-STORE (порт 104 / 4242 / 11112, AE Title) | так аппараты и PACS отправляют снимки |
| C-ECHO | «пинг» DICOM-узла |
| DICOMweb: STOW-RS / WADO-RS / QIDO-RS | HTTP-API, удобно из веба |
| Orthanc REST `POST /instances` | самый быстрый способ залить файл |

## Полезные инструменты

- `pydicom`, `highdicom` (SC / SR / SEG), `pynetdicom`, `SimpleITK`
- Просмотр: Weasis, MicroDicom, 3D Slicer, OHIF
- Консоль DCMTK: `dcmdump файл`, `echoscu host port`, `storescu host port файл`
- Мини-PACS: Orthanc (`docker compose up -d`)
