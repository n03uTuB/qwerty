# -*- coding: utf-8 -*-
"""Широкий поиск геометрических признаков femur_positioning (data-driven)."""
from __future__ import annotations
import os, sys
import numpy as np
import cv2
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src.data import load_manifest
from src.features import _as01, bone_mask, largest_component
from src.dicom_io import read_dicom, to_display_uint8

SEEDS = list(range(10)); N = 5
def clf():
    return Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(max_iter=1000, class_weight="balanced"))])
def cv_auc(X, y, g):
    aucs=[]
    for seed in SEEDS:
        oof=np.full(len(y),np.nan)
        for tr,va in StratifiedGroupKFold(n_splits=N,shuffle=True,random_state=seed).split(X,y,groups=g):
            if len(np.unique(y[tr]))<2: continue
            m=clf().fit(X[tr],y[tr]); oof[va]=m.predict_proba(X[va])[:,1]
        ok=np.isfinite(oof)
        if len(np.unique(y[ok]))<2: continue
        aucs.append(roc_auc_score(y[ok],oof[ok]))
    return float(np.mean(aucs)), float(np.std(aucs))

def feats(arr):
    a = _as01(arr); rows, cols = a.shape
    m = largest_component(bone_mask(a, 0.15), prefer_bottom=True)
    ys, xs = np.nonzero(m)
    d = {}
    if len(ys) < 50:
        return {k:0.0 for k in CANDS}
    y0,y1,x0,x1 = ys.min(),ys.max(),xs.min(),xs.max()
    h,w = rows,cols
    d["f_area"] = float(m.mean())
    d["f_bbox_h"] = float((y1-y0)/h)
    d["f_bbox_w"] = float((x1-x0)/w)
    d["f_aspect"] = float((y1-y0)/max(x1-x0,1))
    d["f_cx"] = float(xs.mean()/w)
    d["f_cy"] = float(ys.mean()/h)
    d["f_cx_off"] = abs(float(xs.mean()/w)-0.5)
    d["f_margin_top"] = float(y0/h)
    d["f_margin_bottom"] = float((h-1-y1)/h)
    d["f_margin_left"] = float(x0/w)
    d["f_margin_right"] = float((w-1-x1)/w)
    d["f_margin_min"] = float(min(y0/h,(h-1-y1)/h,x0/w,(w-1-x1)/w))
    # ориентация minAreaRect
    cnts,_ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ang = 0.0
    if cnts:
        (_,_),(rw,rh),ang = cv2.minAreaRect(max(cnts,key=cv2.contourArea))
        ang = abs(((ang+45)%90)-45)
    d["f_orient"] = float(ang)
    # наклон столба: средний x по строкам нижней половины
    yy = np.arange(y0, y1+1)
    cxrow = np.array([xs[ys==r].mean() if (ys==r).any() else np.nan for r in yy])
    ok = np.isfinite(cxrow)
    slope=0.0
    if ok.sum()>10:
        lo = int(len(yy)*0.5)
        sel = ok.copy(); sel[:lo]=False
        if sel.sum()>5:
            sl = np.polyfit(yy[sel], cxrow[sel], 1)[0]
            slope = float(np.degrees(np.arctan(sl)))
    d["f_shaft_tilt"] = abs(slope)
    # голова: центроид верхней 20% по y
    top = ys < y0 + (y1-y0)*0.25
    d["f_head_dx"] = abs(float(xs[top].mean()/w - xs.mean()/w)) if top.any() else 0.0
    d["f_head_dy"] = float((y1-y0)/h) if top.any() else 0.0
    # solidity / заполненность bbox
    d["f_fill"] = float(m.sum()/max((y1-y0+1)*(x1-x0+1),1))
    # тёмный фон вокруг (коллимация)
    d["f_body_frac"] = float((a>0.12).mean())
    return d

CANDS = ["f_area","f_bbox_h","f_bbox_w","f_aspect","f_cx","f_cy","f_cx_off",
         "f_margin_top","f_margin_bottom","f_margin_left","f_margin_right",
         "f_margin_min","f_orient","f_shaft_tilt","f_head_dx","f_head_dy",
         "f_fill","f_body_frac"]

def main():
    man = load_manifest(C.MANIFEST_CSV)
    man = man[man["region"].isin([C.REGION_FEMUR_LEFT,C.REGION_FEMUR_RIGHT])]
    man = man[man["viol_femur_positioning"].notna()]
    rows=[]; y=[]; g=[]
    for _,r in man.iterrows():
        ds,arr = read_dicom(r["source_path"])
        u8 = to_display_uint8(ds,arr)
        rows.append(feats(u8)); y.append(float(r["viol_femur_positioning"])); g.append(r["study_uid"])
    y=np.array(y); g=np.array(g)
    import pandas as pd
    X=pd.DataFrame(rows)
    res=[]
    for c in CANDS:
        a,s = cv_auc(X[[c]].values, y, g)
        res.append((abs(a-0.5)+0.5,a,s,c))
    res.sort(reverse=True)
    out=["=== кандидаты femur_positioning (AUC±std) ==="]
    for _,a,s,c in res:
        out.append("  %-16s AUC=%.3f±%.3f" % (c,a,s))
    txt="\n".join(out); open("femur_cands.txt","w",encoding="utf-8").write(txt)
    print(txt.encode("ascii","replace").decode())

if __name__=="__main__":
    main()