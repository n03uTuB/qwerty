"""Обмен с PACS: отправка в Orthanc (REST или C-STORE) и приём исследований (C-STORE SCP)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable

import pydicom
import requests
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian


def send_rest(paths: list[Path], url: str = "http://localhost:8042", auth: tuple[str, str] | None = None) -> list[dict]:
    """Orthanc REST API: POST /instances с телом файла. Самый простой способ загрузить результат."""
    results = []
    for path in paths:
        response = requests.post(f"{url.rstrip('/')}/instances", data=Path(path).read_bytes(), auth=auth, timeout=60)
        response.raise_for_status()
        results.append(response.json())
    return results


def send_cstore(paths: list[Path], host: str = "127.0.0.1", port: int = 4242,
                called_ae: str = "ORTHANC", calling_ae: str = "DXAQC") -> list[tuple[Path, int | None]]:
    """Классическая отправка DICOM по сети (так делают аппараты и PACS)."""
    from pynetdicom import AE

    datasets = [(Path(p), pydicom.dcmread(p)) for p in paths]
    ae = AE(ae_title=calling_ae)
    contexts = defaultdict(set)
    for _, ds in datasets:
        contexts[ds.SOPClassUID].add(ds.file_meta.TransferSyntaxUID)
    # Отдельный контекст на каждую пару (SOP Class, syntax): получатель принимает в контексте
    # только один transfer syntax, и сжатые файлы иначе могут остаться без подходящего контекста
    for sop_class, syntaxes in contexts.items():
        for syntax in syntaxes | {ExplicitVRLittleEndian, ImplicitVRLittleEndian}:
            ae.add_requested_context(sop_class, syntax)

    assoc = ae.associate(host, port, ae_title=called_ae)
    if not assoc.is_established:
        raise ConnectionError(f"Не удалось установить ассоциацию с {called_ae}@{host}:{port}")
    statuses = []
    try:
        for path, ds in datasets:
            try:
                status = assoc.send_c_store(ds)
            except ValueError as exc:  # получатель не принял SOP Class или transfer syntax
                print(f"Пропущен {path}: {exc}")
                statuses.append((path, None))
                continue
            statuses.append((path, int(status.Status) if status else None))
    finally:
        assoc.release()
    return statuses


def run_listener(out_dir: str | Path, port: int = 11112, ae_title: str = "DXAQC",
                 on_association: Callable[[list[Path]], None] | None = None) -> None:
    """DICOM-узел: принимает исследования и по завершении ассоциации вызывает обработчик."""
    from pynetdicom import ALL_TRANSFER_SYNTAXES, AE, AllStoragePresentationContexts, evt
    from pynetdicom.sop_class import Verification

    out = Path(out_dir)
    received: dict[int, list[Path]] = defaultdict(list)

    def handle_store(event):
        ds = event.dataset
        ds.file_meta = event.file_meta
        path = out / str(ds.StudyInstanceUID) / f"{ds.SOPInstanceUID}.dcm"
        path.parent.mkdir(parents=True, exist_ok=True)
        ds.save_as(path, enforce_file_format=True)
        received[id(event.assoc)].append(path)
        return 0x0000

    def handle_finished(event):
        paths = received.pop(id(event.assoc), [])
        if paths:
            print(f"Принято файлов: {len(paths)}", flush=True)
            if on_association:
                on_association(paths)

    ae = AE(ae_title=ae_title)
    # По умолчанию принимаются только несжатые синтаксисы — сжатые снимки (JPEG 2000 и т.п.) были бы отклонены
    for context in AllStoragePresentationContexts:
        ae.add_supported_context(context.abstract_syntax, ALL_TRANSFER_SYNTAXES)
    ae.add_supported_context(Verification)
    handlers = [(evt.EVT_C_STORE, handle_store), (evt.EVT_RELEASED, handle_finished), (evt.EVT_ABORTED, handle_finished)]
    print(f"Слушаю DICOM {ae_title}@0.0.0.0:{port}, сохраняю в {out}", flush=True)
    ae.start_server(("0.0.0.0", port), block=True, evt_handlers=handlers)
