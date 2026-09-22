# -*- coding: utf-8 -*-
"""REST API сервиса контроля качества денситометрии (FastAPI).

Эндпоинты:
    GET  /health              — проверка живости сервиса;
    GET  /model/info          — информация о загруженной модели/порогах;
    POST /predict/image       — классификация одного DICOM-файла (upload);
    POST /predict/batch       — пакетная обработка каталога/архива на сервере;
    POST /predict/upload-batch— пакетная обработка загруженных DICOM-файлов.

Сервис полностью локальный: изображения не покидают контур.
"""
from __future__ import annotations

import os
import tempfile
import time
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import config as C
from . import dicom_io
from . import inference

app = FastAPI(title="DXA-QC", version="1.0",
              description="Контроль качества денситометрических исследований")

_STATE = {"controller": None, "thresholds": {}}


def get_controller() -> inference.QualityController:
    if _STATE["controller"] is None:
        _STATE["controller"] = inference.build_controller_from_args(
            device="cuda" if torch.cuda.is_available() else "cpu")
    return _STATE["controller"]


@app.on_event("startup")
def _startup():
    _STATE["thresholds"] = inference.load_thresholds()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model/info")
def model_info():
    ctrl = get_controller()
    return {
        "n_models": len(ctrl.models),
        "device": ctrl.device,
        "quality_threshold": ctrl.quality_threshold,
        "violation_threshold": ctrl.violation_threshold,
        "regions": C.REGIONS,
        "violations": C.VIOLATIONS,
        "hybrid_stacker": bool(getattr(ctrl, "stackers", {})),
        "hybrid_criteria": sorted(getattr(ctrl, "stackers", {}).keys()),
    }


class PredictResponse(BaseModel):
    anatomical_region: str
    quality_class: int
    quality_prob: float
    violation_type: str
    violation_probs: dict
    region_probs: dict


def _row_from_result(region: str, res: dict) -> dict:
    return {
        "anatomical_region": inference.region_label_ru(region),
        "quality_class": res["quality_class"],
        "quality_prob": res["quality_prob"],
        "violation_type": inference.violation_text_ru(res["violation_names"]),
        "violation_probs": res["violation_probs"],
        "region_probs": res["region_probs"],
    }


@app.post("/predict/image", response_model=PredictResponse)
async def predict_image(file: UploadFile = File(...)):
    data = await file.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".dcm")
    tmp.write(data)
    tmp.close()
    try:
        ds, arr = dicom_io.read_dicom(tmp.name)
        if arr is None:
            raise HTTPException(400, "не удалось прочитать пиксельные данные DICOM")
        res = get_controller().predict_array(np.asarray(arr), dicom_io.pixel_spacing(ds, arr))
    finally:
        os.unlink(tmp.name)
    return _row_from_result(res["region"], res)


@app.post("/predict/upload-batch")
async def predict_upload_batch(files: List[UploadFile] = File(...),
                               study_uid: Optional[str] = Form(None)):
    """Пакетная обработка набора загруженных DICOM-файлов."""
    ctrl = get_controller()
    rows = []
    for f in files:
        data = await f.read()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".dcm")
        tmp.write(data)
        tmp.close()
        ts = time.time()
        try:
            ds, arr = dicom_io.read_dicom(tmp.name)
            if arr is None:
                raise ValueError("нет пиксельных данных")
            res = ctrl.predict_array(np.asarray(arr), dicom_io.pixel_spacing(ds, arr))
            row = _row_from_result(res["region"], res)
            row.update(study_uid=study_uid or "", image_uid=str(getattr(ds, "SOPInstanceUID", "")),
                       filename=f.filename, processing_status="Success",
                       time_of_processing=round(time.time() - ts, 3))
        except Exception as e:
            row = dict(anatomical_region="", quality_class=-1, quality_prob=None,
                       violation_type="", violation_probs={}, region_probs={},
                       study_uid=study_uid or "", image_uid="", filename=f.filename,
                       processing_status="Failure", time_of_processing=round(time.time() - ts, 3),
                       error=str(e))
        finally:
            os.unlink(tmp.name)
        rows.append(row)
    return JSONResponse({"n": len(rows), "results": rows})


@app.post("/predict/batch")
def predict_batch(input_dir: str = Form(...), output_csv: str = Form("results.csv")):
    """Пакетная обработка каталога исследований, доступного на сервере."""
    if not os.path.isdir(input_dir):
        raise HTTPException(400, "каталог не найден: %s" % input_dir)
    ctrl = get_controller()
    df = inference.run_batch(input_dir, output_csv, ctrl)
    n_ok = int((df.processing_status == "Success").sum())
    return {"rows": len(df), "success": n_ok, "output_csv": output_csv}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
