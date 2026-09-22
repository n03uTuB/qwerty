"""Командная строка: python -m dxa_sandbox <команда> --help"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path


def _files(paths: list[str]) -> list[Path]:
    from .reading import iter_dicom_files

    return [file for path in paths for file in iter_dicom_files(path)]


def _config(args):
    from .config import DEFAULT

    return replace(DEFAULT, integration=args.integration, model_id=args.model_id, name=args.service_name)


def _add_service_options(parser) -> None:
    parser.add_argument("--integration", choices=["mosmedii", "pum"], default="mosmedii",
                        help="контур ЕРИС: pum — модальность доп. серии ASMT, mosmedii — как у оригинала")
    parser.add_argument("--model-id", type=int, default=1000, help="modelId, выдаётся ЦДТ при подключении")
    parser.add_argument("--service-name", default="DXA-QC", help="идентификатор сервиса в ЕРИС ЕМИАС")


def cmd_samples(args):
    from .samples import SAMPLE_NOTES, make_samples

    paths = make_samples(args.out)
    for name, path in paths.items():
        print(f"{path}  —  {SAMPLE_NOTES.get(name, '')}")


def cmd_inspect(args):
    from .inspection import build_report, print_overview

    rows = build_report(args.path, args.out)
    print_overview(rows)
    print(f"\nОтчёт: {Path(args.out) / 'report.html'}\nТаблица: {Path(args.out) / 'summary.csv'}")


def cmd_dump(args):
    from .inspection import full_dump, private_tags_report
    from .reading import read

    ds = read(args.file, pixels=False)
    print(private_tags_report(ds) if args.private else full_dump(ds))


def cmd_render(args):
    from .reading import extract_encapsulated_document, read, save_png, to_display

    ds = read(args.file)
    if "EncapsulatedDocument" in ds:
        out = extract_encapsulated_document(ds, args.out or Path(args.file).with_suffix(".pdf"))
        print(f"Вложенный документ: {out}")
        return
    out = save_png(to_display(ds, frame=args.frame), args.out or Path(args.file).with_suffix(".png"))
    print(f"PNG: {out}")


def cmd_markup(args):
    import numpy as np

    from .markup import color_markup_masks, horizontal_line_rows, overlay_groups, overlay_mask, remove_markup
    from .reading import read, save_png, to_display

    ds = read(args.file)
    out = Path(args.out)
    image = to_display(ds)
    for group in overlay_groups(ds):
        print(f"overlay {group:04X}: {save_png(overlay_mask(ds, group).astype(np.uint8) * 255, out / f'overlay_{group:04X}.png')}")
    if image.ndim == 3:
        masks = color_markup_masks(image)
        for name, mask in masks.items():
            extra = f", горизонтальные линии y={[round(y) for y in horizontal_line_rows(mask)]}" if name == "green" else ""
            print(f"цвет {name}: {int(mask.sum())} пикс.{extra} -> {save_png(mask.astype(np.uint8) * 255, out / f'color_{name}.png')}")
        if masks:
            clean = remove_markup(image, np.logical_or.reduce(list(masks.values())))
            print(f"без разметки: {save_png(clean, out / 'clean.png')}")
    if not overlay_groups(ds) and image.ndim == 2:
        print("Разметка в overlay и цвете не найдена — ищите её в SR, PDF или приватных тегах (dump --private).")


def _print_study_outputs(outputs: dict, out_dir: Path) -> None:
    for study, files in outputs.items():
        report = json.loads((out_dir / study[-12:] / "qc_report.json").read_text(encoding="utf-8"))
        if "error" in report:
            print(f"Исследование {study}: ОШИБКА {report['error']['category']} — {report['error']['description']}")
            continue
        compliance = report["compliance"]
        print(f"Исследование {study}: {report['verdict']}; дефекты: {', '.join(report['defects']) or 'нет'}; "
              f"замечаний к формату ЦДТ: {len(compliance['problems'])}")
        for problem in compliance["problems"]:
            print(f"  ! {problem}")
        for file in files:
            print(f"  {file}")


def cmd_qc(args):
    from .pipeline import process_path

    _print_study_outputs(process_path(args.path, args.out, _config(args)), Path(args.out))


def cmd_synth(args):
    from collections import Counter

    from .synthetic import generate_dataset

    labels = generate_dataset(args.out, args.n, args.seed)
    counts = Counter(code for label in labels for code in label["defects"])
    print(f"Случаев: {len(labels)}, без нарушений: {sum(1 for l in labels if not l['defects'])}")
    for code, count in sorted(counts.items()):
        print(f"  {code}: {count}")
    print(f"Разметка: {Path(args.out) / 'labels.jsonl'}")


def _evaluate(labels_path: Path, qc_dir: Path, out: Path | None) -> None:
    from .evaluation import collect_predictions, evaluate, format_report, load_labels

    metrics = evaluate(load_labels(labels_path), collect_predictions(qc_dir))
    report = format_report(metrics)
    print(report)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        out.with_suffix(".txt").write_text(report, encoding="utf-8")
        print(f"\nМетрики: {out}")


def cmd_evaluate(args):
    _evaluate(Path(args.labels), Path(args.qc), Path(args.out) if args.out else None)


def cmd_benchmark(args):
    from .pipeline import process_path
    from .synthetic import generate_dataset

    root = Path(args.out)
    generate_dataset(root / "data", args.n, args.seed)
    print(f"Сгенерировано случаев: {args.n}. Прогоняю проверки…", flush=True)
    process_path(root / "data" / "cases", root / "qc", _config(args))
    _evaluate(root / "data" / "labels.jsonl", root / "qc", root / "metrics.json")


def cmd_send(args):
    from .pacs import send_cstore, send_rest

    files = _files(args.paths)
    if args.mode == "rest":
        auth = (args.user, args.password) if args.user else None
        for info in send_rest(files, args.url, auth):
            print(f"{info.get('Status')}: {info.get('ID')}")
        sent = len(files)
    else:
        statuses = send_cstore(files, args.host, args.port, args.called_ae, args.calling_ae)
        for path, status in statuses:
            print(f"0x{status:04X} {path}" if status is not None else f"не отправлен {path}")
        sent = sum(1 for _, status in statuses if status == 0x0000)
    print(f"Успешно отправлено: {sent} из {len(files)}")


def cmd_listen(args):
    from .pacs import run_listener, send_rest
    from .pipeline import process_study

    config = _config(args)

    def on_association(paths):
        if not args.qc:
            return
        # исследование может прийти несколькими ассоциациями — проверяем всё, что уже принято по нему
        for study_dir in sorted({p.parent for p in paths}):
            outputs = process_study(sorted(study_dir.glob("*.dcm")), Path(args.qc_out) / study_dir.name[-12:], config)
            print(f"QC готов: {len(outputs)} файлов", flush=True)
            if args.send_back and outputs:
                send_rest(outputs, args.send_back)
                print(f"Результат отправлен в {args.send_back}", flush=True)

    run_listener(args.out, args.port, args.ae, on_association)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(prog="python -m dxa_sandbox", description="DICOM-песочница для контроля качества DXA")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("samples", help="сгенерировать DICOM с типичными ловушками формата")
    p.add_argument("out", nargs="?", default="data/samples")
    p.set_defaults(func=cmd_samples)

    p = sub.add_parser("inspect", help="разведка папки: HTML-отчёт, CSV, сводка")
    p.add_argument("path")
    p.add_argument("--out", default="out/inspect")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("dump", help="все теги файла")
    p.add_argument("file")
    p.add_argument("--private", action="store_true", help="только приватные теги с превью бинарных значений")
    p.set_defaults(func=cmd_dump)

    p = sub.add_parser("render", help="сохранить PNG (или вложенный PDF)")
    p.add_argument("file")
    p.add_argument("--out")
    p.add_argument("--frame", type=int, default=0)
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("markup", help="извлечь разметку из overlay и цветных пикселей")
    p.add_argument("file")
    p.add_argument("--out", default="out/markup")
    p.set_defaults(func=cmd_markup)

    p = sub.add_parser("qc", help="проверки → доп. серия, SR, JSON по требованиям ЦДТ")
    p.add_argument("path")
    p.add_argument("--out", default="out/qc")
    _add_service_options(p)
    p.set_defaults(func=cmd_qc)

    p = sub.add_parser("synth", help="синтетический набор с нарушениями и разметкой labels.jsonl")
    p.add_argument("out", nargs="?", default="data/synthetic")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("evaluate", help="метрики: labels.jsonl против результатов qc")
    p.add_argument("labels")
    p.add_argument("qc")
    p.add_argument("--out", help="сохранить metrics.json")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("benchmark", help="synth → qc → evaluate одной командой")
    p.add_argument("--out", default="out/benchmark")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    _add_service_options(p)
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("send", help="отправить файлы в Orthanc")
    p.add_argument("paths", nargs="+")
    p.add_argument("--mode", choices=["rest", "cstore"], default="rest")
    p.add_argument("--url", default="http://localhost:8042")
    p.add_argument("--user")
    p.add_argument("--password")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=4242)
    p.add_argument("--called-ae", default="ORTHANC")
    p.add_argument("--calling-ae", default="DXAQC")
    p.set_defaults(func=cmd_send)

    p = sub.add_parser("listen", help="DICOM-узел: принимать исследования (и сразу проверять)")
    p.add_argument("--port", type=int, default=11112)
    p.add_argument("--ae", default="DXAQC")
    p.add_argument("--out", default="data/incoming")
    p.add_argument("--qc", action="store_true", help="запускать QC после приёма")
    p.add_argument("--qc-out", default="out/qc")
    p.add_argument("--send-back", metavar="URL", help="отправить результат в Orthanc, напр. http://localhost:8042")
    _add_service_options(p)
    p.set_defaults(func=cmd_listen)

    args = parser.parse_args(argv)
    args.func(args)
    return 0
