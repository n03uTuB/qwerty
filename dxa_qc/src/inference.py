# -*- coding: utf-8 -*-
"""Пакетный инференс: DICOM-исследования -> структурированный отчёт.

Формирует таблицу с колонками из ТЗ:
    path_to_study, study_uid, image_uid, anatomical_region,
    quality_class, violation_type, processing_status, time_of_processing
а также (дополнительно) вероятности и список сработавших критериев.
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
from . import model as model_lib

# Порог качества по умолчанию (подбирается при валидации, см. calibrate.py)
DEFAULT_QUALITY_THRESHOLD = 0.5
DEFAULT_VIOLATION_THRESHOLD = 0.5

REGION_LABEL_RU = {
    C.REGION_SPINE: "lumbar_spine",
    C.REGION_FEMUR_LEFT: "proximal_femur_left",
    C.REGION_FEMUR_RIGHT: "proximal_femur_right",
}


def region_label_ru(region: str) -> str:
    """Название области в формате организатора (без стороны)."""
    return C.REGION_LABEL_RU.get(region, region)


def violation_text_ru(names: List[str]) -> str:
    """Строка violation_type по закрытому списку организатора.

    Несколько нарушений перечисляются через «; », при отсутствии — пусто.
    """
    labels = []
    for n in names:
        lab = C.VIOLATION_LABEL_RU.get(n)
        if lab and lab not in labels:      # без дублей (укладка общая для областей)
            labels.append(lab)
    return C.VIOLATION_SEP.join(labels) if labels else C.VIOLATION_EMPTY


class QualityController:
    """Обёртка над ансамблем моделей для инференса одного изображения."""

    def __init__(self, weights: Optional[List[str]] = None,
                 device: str = "cpu",
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
        # пороги по областям/критериям (переопределяют скалярные)
        self.quality_thresholds = quality_thresholds or {}
        self.violation_thresholds = violation_thresholds or {}
        self.models = []
        for w in weights:
            if not os.path.isfile(w):
                continue
            m = model_lib.build_model(pretrained=False)
            model_lib.load_weights(m, w, device=device)
            m.to(device)
            self.models.append(m)
        if not self.models:
            raise FileNotFoundError(
                "Не найдено ни одного файла весов. Сначала обучите модель "
                "(python -m src.train).")

        # --- гибридный стекер (CNN + геометрические признаки), если обучен ---
        self.stackers = {}
        stack_path = os.path.join(C.ARTIFACTS_DIR, "stacker.joblib")
        if os.path.isfile(stack_path):
            try:
                import joblib

                from . import features as feat_lib

                self.stackers = joblib.load(stack_path)
                self._feature_fn = feat_lib.compute_features
            except Exception as e:
                print("[inference] стекер не загружен:", e)
                self.stackers = {}

    @torch.no_grad()
    def predict_array(self, arr: np.ndarray, spacing=None, copies: float = 1.0) -> Dict:
        img = dicom_io.preprocess(arr, self.size)
        x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(self.device)

        region = dicom_io.detect_region(arr)          # геометрический детектор
        ridx = C.REGION_TO_IDX[region]

        q_probs, v_probs, r_probs = [], [], []
        for m in self.models:
            q, v, r = model_lib.predict_probs(m, x, tta=self.tta)
            q_probs.append(q.cpu().numpy()[0])
            v_probs.append(v.cpu().numpy()[0])
            r_probs.append(r.cpu().numpy()[0])
        q = np.mean(q_probs, axis=0)
        v = np.mean(v_probs, axis=0)
        r = np.mean(r_probs, axis=0)

        # --- гибридная коррекция (CNN + геометрия) ---
        feat = None
        if self.stackers:
            # масштаб снимка обязателен: от него зависят углы (см. dicom_io.pixel_spacing)
            feat = self._feature_fn(np.asarray(arr), spacing or C.PIXEL_SPACING_MM,
                                    region=region, copies=copies)

        q_region = float(q[ridx])
        if feat is not None and "quality" in self.stackers:
            st = self.stackers["quality"]
            x = np.array([q_region] + [feat[f] for f in st["features"]]).reshape(1, -1)
            q_region = float(st["clf"].predict_proba(x)[0, 1])

        v_fused = v.copy()
        if feat is not None:
            for name in C.VIOLATIONS:
                if name not in self.stackers:
                    continue
                st = self.stackers[name]
                i = C.VIOLATION_IDX[name]
                x = np.array([v[i]] + [feat[f] for f in st["features"]]).reshape(1, -1)
                v_fused[i] = float(st["clf"].predict_proba(x)[0, 1])

        # список сработавших нарушений только для критериев данной области
        crit = C.REGION_CRITERIA[region]
        fired = []
        for name in crit:
            i = C.VIOLATION_IDX[name]
            thr = self.violation_thresholds.get(name, self.violation_threshold)
            if v_fused[i] >= thr:
                fired.append(name)

        # quality_class = ИЛИ(сработавших критериев). Гейт по CNN-качеству снят:
        # в разметке организатора quality_class практически совпадает с ИЛИ
        # критериев (246/249), а гейт лишь терял true positive (macro-F1
        # организатора 0.302 -> 0.407, ROC-AUC качества 0.611 -> 0.707).
        quality_class = int(bool(fired))
        # вероятность качества — максимум по критериям области (вариант B,
        # лучший по balanced accuracy / macro-F1 / ROC-AUC организатора)
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
        return self.predict_array(np.asarray(arr), dicom_io.pixel_spacing(ds, arr))


def process_study(controller: QualityController, study_uid: str, study_path: str) -> List[dict]:
    """Обработать одно исследование -> список строк отчёта."""
    rows: List[dict] = []
    t0 = time.time()
    try:
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

    # Исследование есть, но ни один файл не читается (битые/пустые DICOM): по ТЗ
    # такое исследование должно попасть в отчёт со статусом Failure, а батч — продолжиться.
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
                                           copies=copies)
            status = "Success"
            viol_text = violation_text_ru(res["violation_names"])
            row = dict(
                path_to_study=study_path,
                study_uid=study_uid,
                image_uid=image_uid,
                anatomical_region=region_label_ru(res["region"]),
                quality_class=res["quality_class"],
                violation_type=viol_text,
                processing_status=status,
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

    for study_uid, study_path in dicom_io.iter_studies(studies_root):
        rows = process_study(controller, study_uid, study_path)
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
    path = os.path.join(C.ARTIFACTS_DIR, "thresholds.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def build_controller_from_args(weights: Optional[List[str]] = None,
                               device: str = "cpu",
                               quality_threshold: Optional[float] = None,
                               violation_threshold: Optional[float] = None,
                               size: int = C.IMAGE_SIZE,
                               tta: bool = C.USE_TTA) -> QualityController:
    if weights is None:
        # ансамбль: финальная модель + все фолды, если есть
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
        violation_threshold = float(sum(vals) / len(vals)) if vals else DEFAULT_VIOLATION_THRESHOLD

    return QualityController(weights, device=device,
                             quality_threshold=quality_threshold,
                             violation_threshold=violation_threshold, size=size,
                             quality_thresholds=qbr, violation_thresholds=vth,
                             tta=tta)
