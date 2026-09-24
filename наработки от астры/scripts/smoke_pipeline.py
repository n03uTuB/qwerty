# -*- coding: utf-8 -*-
"""Дымовой прогон всего пайплайна на синтетическом датасете.

Строит ~100 правдоподобных DXA-подобных PNG (позвоночник + бедро), манифест с
метками, атлас-шаблон, признаки, затем последовательно запускает
train -> stack -> calibrate -> evaluate. Реальных данных не требует.

    python scripts/smoke_pipeline.py
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config as C  # noqa: E402
from src.data import synthetic as syn  # noqa: E402
from src.data.manifest import save_manifest  # noqa: E402

RNG = np.random.default_rng(7)
ROWS, COLS = 300, 320


def _finish(img, rows=ROWS, cols=COLS):
    img = np.clip(img + RNG.normal(0, 0.01, img.shape), 0, 1)
    return (img * 255).astype(np.uint8)


def base_spine(shift=0.0, angle=0.0, width=1.0):
    import cv2
    img = np.zeros((ROWS, COLS), np.float32)
    cx = COLS // 2 + int(shift * COLS)
    yy, xx = np.mgrid[0:ROWS, 0:COLS]
    body = (((xx - cx) / (COLS * 0.42)) ** 2 + ((yy - ROWS / 2) / (ROWS * 0.55)) ** 2) < 1
    img[body] = RNG.uniform(0.10, 0.20)
    col = (np.abs(xx - cx) < COLS * 0.055 * width) & (yy > ROWS * 0.12) & (yy < ROWS * 0.92)
    img[col] = RNG.uniform(0.75, 0.95)
    iliac = (np.abs(xx - cx) < COLS * 0.30) & (yy > ROWS * 0.82)
    img[iliac] = np.maximum(img[iliac], RNG.uniform(0.55, 0.75))
    for k in range(5):
        y0 = int(ROWS * (0.18 + 0.14 * k))
        img[y0:y0 + 8, cx - int(COLS * 0.075):cx + int(COLS * 0.075)] = RNG.uniform(0.85, 1.0)
    if angle:
        m = cv2.getRotationMatrix2D((cx, ROWS / 2), angle, 1.0)
        img = cv2.warpAffine(img, m, (COLS, ROWS), borderMode=cv2.BORDER_CONSTANT)
    return _finish(img)


def base_femur(margin=0.30):
    import cv2
    img = np.zeros((ROWS, COLS), np.float32)
    yy, xx = np.mgrid[0:ROWS, 0:COLS]
    body = (((xx - COLS / 2) / (COLS * 0.40)) ** 2 + ((yy - ROWS / 2) / (ROWS * 0.55)) ** 2) < 1
    img[body] = RNG.uniform(0.10, 0.20)
    # диафиз под углом + шейка + головка
    shaft = (np.abs((xx - COLS * 0.5) - (yy - ROWS) * 0.25) < COLS * 0.05) & (yy > ROWS * 0.45)
    img[shaft] = RNG.uniform(0.70, 0.90)
    neck = (np.abs((xx - COLS * 0.5) - (yy - ROWS * 0.45) * 0.7) < COLS * 0.05) & (yy > ROWS * 0.30) & (yy < ROWS * 0.55)
    img[neck] = RNG.uniform(0.70, 0.90)
    head = ((xx - COLS * 0.62) ** 2 + (yy - ROWS * 0.28) ** 2) < (COLS * 0.10) ** 2
    img[head] = RNG.uniform(0.80, 0.95)
    troch = ((xx - COLS * 0.40) ** 2 + (yy - ROWS * 0.45) ** 2) < (COLS * 0.09) ** 2
    img[troch] = np.maximum(img[troch], RNG.uniform(0.65, 0.85))
    # укладка: сдвиг кости к краю
    if margin < 0.25:
        m = np.float32([[1, 0, -int(COLS * (0.25 - margin))], [0, 1, 0]])
        img = cv2.warpAffine(img, m, (COLS, ROWS), borderMode=cv2.BORDER_CONSTANT)
    return _finish(img)


def _mk_row(uid, study, region, img, quality, viols):
    cache_dir = C.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, uid + ".png")
    Image.fromarray(img).save(path)
    row = dict(study_uid=study, image_uid=uid, source_path=path, cache_path=path,
               region=region, region_idx=C.REGION_TO_IDX[region],
               rows=img.shape[0], cols=img.shape[1], spacing_x=0.6, spacing_y=0.6,
               copies=1, quality=quality, synthetic=False)
    # критерии области: 0/1 (как в реальной разметке); чужие критерии -> NaN
    region_criteria = C.REGION_CRITERIA[region]
    for name in C.VIOLATIONS:
        if name in region_criteria:
            row["viol_" + name] = float(viols.get(name, 0.0))
            row["mask_" + name] = 1.0
        else:
            row["viol_" + name] = np.nan
            row["mask_" + name] = 0.0
    return row


def build_dataset():
    rows = []
    n = 0
    # --- позвоночник ---
    plan = (["clean"] * 34 + ["axis"] * 10 + ["positioning"] * 10 + ["artifact"] * 16)
    for i, kind in enumerate(plan):
        study = "S_spine_%03d" % (i // 2)
        uid = "spine_%03d" % i
        if kind == "clean":
            img = base_spine()
            v, q = {}, 0.0
        elif kind == "axis":
            img = base_spine(angle=RNG.uniform(9, 16))
            v, q = {"spine_axis": 1.0}, 1.0
        elif kind == "positioning":
            img = base_spine(shift=RNG.uniform(0.10, 0.20))
            v, q = {"spine_positioning": 1.0}, 1.0
        else:
            img = syn.edge_object(base_spine(), RNG)
            v, q = {"spine_artifacts": 1.0}, 1.0
        rows.append(_mk_row(uid, study, C.REGION_SPINE, img, q, v))
        n += 1
    # --- бедро ---
    fplan = (["clean"] * 24 + ["positioning"] * 8 + ["roi"] * 8)
    for i, kind in enumerate(fplan):
        region = C.REGION_FEMUR_LEFT if i % 2 == 0 else C.REGION_FEMUR_RIGHT
        study = "S_femur_%03d" % (i // 2)
        uid = "femur_%03d" % i
        if kind == "clean":
            img = base_femur(margin=0.30)
            v, q = {}, 0.0
        elif kind == "positioning":
            img = base_femur(margin=RNG.uniform(0.05, 0.12))
            v, q = {"femur_positioning": 1.0}, 1.0
        else:
            img = base_femur(margin=0.30)
            v, q = {"femur_roi": 1.0}, 1.0
        rows.append(_mk_row(uid, study, region, img, q, v))
        n += 1
    df = pd.DataFrame(rows)
    save_manifest(df, C.MANIFEST_CSV)
    print("[smoke] изображений:", len(df))
    return df


def run(*args):
    cmd = [sys.executable, "-m"] + list(args)
    print("\n>>>", " ".join(cmd))
    r = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if r.returncode != 0:
        raise SystemExit("FAILED: %s" % " ".join(args))


def main():
    build_dataset()
    from src.atlas import Atlas, build_atlas
    from src.data import load_manifest
    build_atlas(load_manifest(C.MANIFEST_CSV))
    assert Atlas.load() is not None, "атлас не построен"

    run("src.train", "--folds", "2", "--epochs", "1", "--size", "128",
        "--batch", "8", "--no-tta")
    run("src.stack")
    run("src.calibrate")
    run("src.evaluate", "--thr", "f1", "--stacked")
    print("\nPIPELINE_OK")


if __name__ == "__main__":
    main()