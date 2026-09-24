# -*- coding: utf-8 -*-
"""Пакетный инференс: DICOM-исследования -> структурированный отчёт.

Колонки по ТЗ: path_to_study, study_uid, image_uid, anatomical_region,
quality_class, violation_type, processing_status, time_of_processing
(+ quality_prob, violation_probs для аудита/калибровки).

Ключевые отличия гибрида:
  * бэкбон каждой модели ансамбля определяется АВТОМАТИЧЕСКИ по её весам
    (:func:`src.models.infer_backbone_from_weights`) — исключает молчаливую
    загрузку resnet34-весов в resnet18 (баг fable_solution);
  * пиксельный конвейер совпадает с обучением (modality/VOI LUT + костное окно);
  * гибридный стекер (CNN + геометрия) применяется к вероятностям критериев
    с корректным учётом флага ``use_cnn``;
  * любой сбой изображения -> ``processing_status='Failure'`` (try/except),
    батч при этом продолжается.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from . import config as C
from . import dicom_io
from .models import build_model, infer_backbone_from_weights, load_weights, predict_probs

DEFAULT_QUALITY_THRESHOLD = 0.5
DEFAULT_VIOLATION_THRESHOLD = 0.5


def region_label_ru(region: str) -> str:
    """Название области в формате организатора."""
    return C.REGION_LABEL_RU.get(region, region)


def violation_text_ru(names: List[str]) -> str:
    """Строка violation_type по закрытому списку (через «; », пусто если нет)."""
    labels = []
    for n in names:
        lab = C.VIOLATION_LABEL_RU.get(n)
        if lab and lab not in labels:      # без дублей (укладка общая для областей)
            labels.append(lab)
    return C.VIOLATION_SEP.join(labels) if labels else C.VIOLATION_EMPTY


class QualityController:
    """Обёртка над ансамблем моделей для инференса одного изображения."""

    def __init__(self, weights: Optional[List[str]] = None, device: str = "cpu",
                 quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
                 violation_threshold: float = DEFAULT_VIOLATION_THRESHOLD,
                 size: int = C.IMAGE_SIZE,
                 quality_thresholds: Optional[Dict[str, float]] = None,
                 violation_thresholds: Optional[Dict[str, float]] = None,
                 tta: bool = C.USE_TTA):
        if weights is None:
            weights = [C.BEST_WEIGHTS]
        self.device = device
        self.size = size
        self.tta = tta
        self.quality_threshold = quality_threshold
        self.violation_threshold = violation_threshold
        self.quality_thresholds = quality_thresholds or {}
        self.violation_thresholds = violation_thresholds or {}
        self.backbones: List[str] = []
        self.models = []
        for w in weights:
            if not os.path.isfile(w):
                continue
            # автоопределение архитектуры по весам (иначе — дефолт)
            bb = infer_backbone_from_weights(w, device=device) or C.BACKBONE
            m = build_model(pretrained=False, backbone=bb)
            load_weights(m, w, device=device)
            m.to(device)
            self.models.append(m)
            self.backbones.append(bb)
        if not self.models:
            raise FileNotFoundError(
                "Не найдено ни одного файла весов. Сначала обучите модель "
                "(python -m src.train).")
        print(f"[inference] ансамбль: {len(self.models)} моделей, "
              f"бэкбоны: {self.backbones}")

        self.stackers: Dict = {}
        if os.path.isfile(C.STACKER_PATH):
            try:
                import joblib

                from . import features as feat_lib

                self.stackers = joblib.load(C.STACKER_PATH)
                self._feature_fn = feat_lib.compute_features
            except Exception as e:
                print("[inference] стекер не загружен:", e)
                self.stackers = {}

    @torch.no_grad()
    def predict_array(self, arr: np.ndarray, spacing=None, copies: float = 1.0,
                      ds=None) -> Dict:
        # ds -> тот же вход, что и при обучении (modality/VOI LUT + bone window)
        img = dicom_io.preprocess(arr, self.size, ds=ds)
        x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(self.device)

        region = dicom_io.detect_region(arr)          # геометрический детектор
        ridx = C.REGION_TO_IDX[region]

        q_probs, v_probs, r_probs = [], [], []
        for m in self.models:
            q, v, r = predict_probs(m, x, tta=self.tta)
            q_probs.append(q.cpu().numpy()[0])
            v_probs.append(v.cpu().numpy()[0])
            r_probs.append(r.cpu().numpy()[0])
        q = np.mean(q_probs, axis=0)
        v = np.mean(v_probs, axis=0)
        r = np.mean(r_probs, axis=0)

        feat = None
        if self.stackers:
            feat = self._feature_fn(np.asarray(arr), spacing or C.PIXEL_SPACING_MM,
                                    region=region, copies=copies)

        q_region = float(q[ridx])
        if feat is not None and "quality" in self.stackers:
            st = self.stackers["quality"]
            xq = np.array([q_region] + [feat[f] for f in st["features"]]).reshape(1, -1)
            q_region = float(st["clf"].predict_proba(xq)[0, 1])

        v_fused = v.copy()
        if feat is not None:
            for name in C.VIOLATIONS:
                if name not in self.stackers:
                    continue
                st = self.stackers[name]
                i = C.VIOLATION_IDX[name]
                feats = [feat[f] for f in st["features"]]
                if st.get("use_cnn", True):     # True — гибрид, False — чистая геометрия
                    feats = [v[i]] + feats
                xv = np.array(feats).reshape(1, -1)
                v_fused[i] = float(st["clf"].predict_proba(xv)[0, 1])

        crit = C.REGION_CRITERIA[region]
        fired = []
        for name in crit:
            i = C.VIOLATION_IDX[name]
            thr = self.violation_thresholds.get(name, self.violation_threshold)
            if v_fused[i] >= thr:
                fired.append(name)

        # quality_class = ИЛИ(сработавших критериев) — совпадает с разметкой (246/249)
        quality_class = int(bool(fired))
        quality_prob = (max(float(v_fused[C.VIOLATION_IDX[n]]) for n in crit)
                        if crit else float(q_region))

        return dict(
            region=region,
            quality_prob=quality_prob,
            quality_class=quality_class,
            violation_probs={n: float(v_fused[C.VIOLATION_IDX[n]]) for n in C.VIOLATIONS},
            violation_names=fired,
            region_probs={C.REGIONS[i]: float(r[i]) for i in range(len(C.REGIONS))},
        )

    def predict_image(self, path: str) -> Dict:
        ds, arr = dicom_io.read_dicom(path)
        if arr is None:
            raise ValueError("не удалось прочитать пиксельные данные: %s" % path)
        return self.predict_array(np.asarray(arr), dicom_io.pixel_spacing(ds, arr),
                                  ds=ds)


def process_study(controller: QualityController, study_uid: str,
                  study_path: str,
                  files: Optional[List[str]] = None) -> List[dict]:
    """Обработать одно исследование -> список строк отчёта.

    ``files`` — явный список DICOM (для «плоского» каталога, где файлы лежат без
    подпапок); если не задан, ищутся рекурсивно в ``study_path``.
    """
    rows: List[dict] = []
    t0 = time.time()
    try:
        if files is None:
            files = dicom_io.find_dicom_files(study_path)
        uniq = dicom_io.dedupe_unique_images(files)
        copies_map = dicom_io.study_copies(files)
    except Exception:
        return [dict(
            path_to_study=study_path, study_uid=study_uid, image_uid="",
            anatomical_region="", quality_class=-1, violation_type="",
            processing_status="Failure", time_of_processing=round(time.time() - t0, 3),
            error=traceback.format_exc(limit=1),
        )]

    if not uniq:
        return [dict(
            path_to_study=study_path, study_uid=study_uid, image_uid="",
            anatomical_region="", quality_class=-1, violation_type="",
            processing_status="Failure", time_of_processing=round(time.time() - t0, 3),
            error="не найдено читаемых DICOM-изображений",
        )]

    for path, ds, arr in uniq:
        ts = time.time()
        image_uid = str(getattr(ds, "SOPInstanceUID", "") or "")
        try:
            import hashlib as _hl
            px_hash = _hl.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()
            copies = float(copies_map.get(px_hash, 1))
            res = controller.predict_array(np.asarray(arr),
                                           dicom_io.pixel_spacing(ds, arr),
                                           copies=copies, ds=ds)
            row = dict(
                path_to_study=study_path,
                study_uid=study_uid,
                image_uid=image_uid,
                anatomical_region=region_label_ru(res["region"]),
                quality_class=res["quality_class"],
                violation_type=violation_text_ru(res["violation_names"]),
                processing_status="Success",
                time_of_processing=round(time.time() - ts, 3),
                quality_prob=round(res["quality_prob"], 4),
                violation_probs=";".join(
                    "%s=%.3f" % (n, res["violation_probs"][n]) for n in C.VIOLATIONS),
            )
        except Exception:
            row = dict(
                path_to_study=study_path, study_uid=study_uid, image_uid=image_uid,
                anatomical_region="", quality_class=-1, violation_type="",
                processing_status="Failure",
                time_of_processing=round(time.time() - ts, 3),
                quality_prob=None, violation_probs="",
                error=traceback.format_exc(limit=1),
            )
        rows.append(row)
    return rows


def run_batch(input_root: str, output_csv: str, controller: QualityController,
              output_xlsx: Optional[str] = None) -> pd.DataFrame:
    """Пакетная обработка всех исследований в каталоге input_root."""
    all_rows: List[dict] = []
    if os.path.isdir(os.path.join(input_root, "исследования")):
        studies_root = os.path.join(input_root, "исследования")
    else:
        studies_root = input_root

    studies = list(dicom_io.iter_studies(studies_root))
    if studies:
        # обычная раскладка: подпапка = исследование
        groups = [(uid, path, None) for uid, path in studies]
    else:
        # «плоский» каталог: DICOM-файлы лежат прямо в корне.
        # Группируем по StudyInstanceUID (иначе — всё как одно исследование).
        flat = dicom_io.find_dicom_files(studies_root)
        by_uid: Dict[str, List[str]] = {}
        for fp in flat:
            try:
                ds, _arr = dicom_io.read_dicom(fp)
                uid = str(getattr(ds, "StudyInstanceUID", "") or "")
            except Exception:
                uid = ""
            if not uid:
                uid = os.path.basename(os.path.abspath(studies_root))
            by_uid.setdefault(uid, []).append(fp)
        groups = [(uid, studies_root, fl) for uid, fl in sorted(by_uid.items())]

    for study_uid, study_path, files in groups:
        rows = process_study(controller, study_uid, study_path, files=files)
        all_rows.extend(rows)
        print("  %s -> %d изображений" % (study_uid[:40], len(rows)))

    df = pd.DataFrame(all_rows)
    cols = ["path_to_study", "study_uid", "image_uid", "anatomical_region",
            "quality_class", "violation_type", "processing_status",
            "time_of_processing", "quality_prob", "violation_probs"]
    for c in cols:
        if c not in df:
            df[c] = None
    df = df[cols + [c for c in df.columns if c not in cols]]

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    if output_xlsx:
        try:
            df.to_excel(output_xlsx, index=False)
        except Exception as e:
            print("  [warn] xlsx не сохранён:", e)
    return df


def load_thresholds() -> dict:
    if os.path.isfile(C.THRESHOLDS_JSON):
        with open(C.THRESHOLDS_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {}


def build_controller_from_args(weights: Optional[List[str]] = None,
                               device: str = "cpu",
                               quality_threshold: Optional[float] = None,
                               violation_threshold: Optional[float] = None,
                               size: int = C.IMAGE_SIZE,
                               tta: bool = C.USE_TTA) -> QualityController:
    if weights is None:
        weights = []
        if os.path.isfile(C.BEST_WEIGHTS):
            weights.append(C.BEST_WEIGHTS)
        for k in range(C.N_FOLDS):
            fp = C.FOLD_WEIGHTS_TMPL.format(k=k)
            if os.path.isfile(fp):
                weights.append(fp)

    thr = load_thresholds()
    qbr = {r: v["threshold"] for r, v in thr.get("quality_by_region", {}).items()
           if v.get("threshold") is not None}
    vth = {n: v["threshold"] for n, v in thr.get("violations", {}).items()
           if v.get("threshold") is not None}

    if quality_threshold is None:
        quality_threshold = thr.get("quality", {}).get("threshold",
                                                        DEFAULT_QUALITY_THRESHOLD)
    if violation_threshold is None:
        vals = list(vth.values())
        violation_threshold = (float(sum(vals) / len(vals)) if vals
                               else DEFAULT_VIOLATION_THRESHOLD)

    return QualityController(weights, device=device,
                             quality_threshold=quality_threshold,
                             violation_threshold=violation_threshold, size=size,
                             quality_thresholds=qbr, violation_thresholds=vth,
                             tta=tta)