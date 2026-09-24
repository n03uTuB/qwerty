# -*- coding: utf-8 -*-
"""Поиск пар/троек признаков femur_positioning: подбор на сидах 0-4, проверка 5-9."""
from __future__ import annotations
import os, sys, itertools
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src.data import load_manifest
from src.features import load_features, _as01, bone_mask, largest_component
from src.dicom_io import read_dicom, to_display_uint8

def clf():
    return Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(max_iter=1000, class_weight="balanced"))])
def cv(X, y, g, cnn, seeds):
    aucs=[]
    for seed in seeds:
        oof=np.full(len(y),np.nan)
        for tr,va in StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed).split(X,y,groups=g):
            if len(np.unique(y[tr]))<2: continue
            Xf=np.column_stack([cnn,X]); m=clf().fit(Xf[tr],y[tr]); oof[va]=m.predict_proba(Xf[va])[:,1]
        ok=np.isfinite(oof)
        if len(np.unique(y[ok]))<2: continue
        aucs.append(roc_auc_score(y[ok],oof[ok]))
    return float(np.mean(aucs))
def f1cv(X,y,g,cnn,seeds):
    fs=[]
    for seed in seeds:
        oof=np.full(len(y),np.nan)
        for tr,va in StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed).split(X,y,groups=g):
            if len(np.unique(y[tr]))<2: continue
            Xf=np.column_stack([cnn,X]); m=clf().fit(Xf[tr],y[tr]); oof[va]=m.predict_proba(Xf[va])[:,1]
        ok=np.isfinite(oof)
        fs.append(f1_score(y[ok],(oof[ok]>=0.5).astype(int),zero_division=0))
    return float(np.mean(fs)), float(np.std(fs))

def extra(arr):
    a=_as01(arr); rows,cols=a.shape
    m=largest_component(bone_mask(a,0.15),prefer_bottom=True)
    ys,xs=np.nonzero(m); d={}
    if len(ys)<50:
        return {k:0.0 for k in ["f_head_dx","f_bbox_h","f_cx_off","f_fill","f_orient"]}
    y0,y1,x0,x1=ys.min(),ys.max(),xs.min(),xs.max()
    top=ys<y0+(y1-y0)*0.25
    d["f_head_dx"]=abs(float(xs[top].mean()/cols-xs.mean()/cols)) if top.any() else 0.0
    d["f_bbox_h"]=float((y1-y0)/rows)
    d["f_cx_off"]=abs(float(xs.mean()/cols)-0.5)
    d["f_fill"]=float(m.sum()/max((y1-y0+1)*(x1-x0+1),1))
    import cv2
    cnts,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    ang=0.0
    if cnts:
        (_,_),(_,_),ang=cv2.minAreaRect(max(cnts,key=cv2.contourArea))
        ang=abs(((ang+45)%90)-45)
    d["f_orient"]=float(ang)
    return d

def main():
    man=load_manifest(C.MANIFEST_CSV); df=load_features(man)
    oof_v=np.load(C.OOF_VIOLATION_NPY); j=C.VIOLATION_IDX["femur_positioning"]
    m=df["region"].isin([C.REGION_FEMUR_LEFT,C.REGION_FEMUR_RIGHT]).values & df["viol_femur_positioning"].notna().values
    sub=df[m].reset_index(drop=True); idx=df.index[m].values
    y=sub["viol_femur_positioning"].astype(float).values; g=sub["study_uid"].values
    cnn=oof_v[idx,j]
    exdf=pd.DataFrame([extra(to_display_uint8(*read_dicom(r["source_path"]))) for _,r in sub.iterrows()])
    base=["femur_margin_min_cm","femur_width_cm","femur_trochanter_bulge",
          "femur_shaft_deg","femur_axis_angle","femur_margin_top_cm",
          "femur_neck_width_mm","fatlas_resid_bone","fatlas_bone_iou","falign_angle"]
    X=pd.concat([sub[base].reset_index(drop=True), exdf],axis=1)
    TR=list(range(5)); TE=list(range(5,10))
    cols=list(X.columns)
    cur=["femur_margin_min_cm","femur_width_cm"]
    a_cur_tr=cv(X[cur].values,y,g,cnn,TR); a_cur_te=cv(X[cur].values,y,g,cnn,TE)
    print("текущий %s: train=%.3f heldout=%.3f"%(cur,a_cur_tr,a_cur_te))
    # одиночные
    singles=sorted(((cv(X[[c]].values,y,g,cnn,TR),c) for c in cols),reverse=True)
    top=[c for _,c in singles[:9]]
    print("топ-9 одиночных:", ["%s=%.3f"%(c,a) for a,c in singles[:9]])
    results=[]
    # пары и тройки из топ-9
    for r in (2,3):
        for combo in itertools.combinations(top,r):
            atr=cv(X[list(combo)].values,y,g,cnn,TR)
            results.append((atr,list(combo)))
    results.sort(reverse=True)
    print("\nтоп-12 по train (проверка на held-out):")
    out=[]
    for atr,combo in results[:12]:
        ate=cv(X[combo].values,y,g,cnn,TE)
        f1m,f1s=f1cv(X[combo].values,y,g,cnn,TE)
        out.append((ate,combo,atr,f1m,f1s))
        print("  train=%.3f held=%.3f F1=%.3f±%.3f  %s"%(atr,ate,f1m,f1s,combo))
    out.sort(reverse=True)
    best=out[0]
    txt="ЛУЧШИЙ по held-out: %s\n  train=%.3f held-out AUC=%.3f F1=%.3f±%.3f"%(best[1],best[2],best[0],best[3],best[4])
    txt+="\nтекущий: train=%.3f held-out=%.3f"%(a_cur_tr,a_cur_te)
    open("femur_best2.txt","w",encoding="utf-8").write(txt)
    print("\n"+txt.encode("ascii","replace").decode())

if __name__=="__main__":
    main()