# -*- coding: utf-8 -*-
"""Честный отбор признаков femur_positioning: подбор на сидах 0-4, проверка на 5-9."""
from __future__ import annotations
import os, sys, itertools
import numpy as np, pandas as pd, cv2
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

def cv(X, y, g, cnn, seeds, mode="fused"):
    aucs=[]
    for seed in seeds:
        oof=np.full(len(y),np.nan)
        for tr,va in StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed).split(X,y,groups=g):
            if len(np.unique(y[tr]))<2: continue
            Xf = np.column_stack([cnn,X]) if mode=="fused" else X
            m=clf().fit(Xf[tr],y[tr]); oof[va]=m.predict_proba(Xf[va])[:,1]
        ok=np.isfinite(oof)
        if len(np.unique(y[ok]))<2: continue
        aucs.append(roc_auc_score(y[ok],oof[ok]))
    return float(np.mean(aucs))

def extra(arr):
    a=_as01(arr); rows,cols=a.shape
    m=largest_component(bone_mask(a,0.15),prefer_bottom=True)
    ys,xs=np.nonzero(m); d={}
    if len(ys)<50:
        return {k:0.0 for k in ["f_head_dx","f_bbox_h","f_cx_off","f_fill"]}
    y0,y1,x0,x1=ys.min(),ys.max(),xs.min(),xs.max()
    top=ys<y0+(y1-y0)*0.25
    d["f_head_dx"]=abs(float(xs[top].mean()/cols-xs.mean()/cols)) if top.any() else 0.0
    d["f_bbox_h"]=float((y1-y0)/rows)
    d["f_cx_off"]=abs(float(xs.mean()/cols)-0.5)
    d["f_fill"]=float(m.sum()/max((y1-y0+1)*(x1-x0+1),1))
    return d

def main():
    man=load_manifest(C.MANIFEST_CSV)
    df=load_features(man)
    oof_v=np.load(C.OOF_VIOLATION_NPY); j=C.VIOLATION_IDX["femur_positioning"]
    m=df["region"].isin([C.REGION_FEMUR_LEFT,C.REGION_FEMUR_RIGHT]).values & df["viol_femur_positioning"].notna().values
    sub=df[m].reset_index(drop=True); idx=df.index[m].values
    y=sub["viol_femur_positioning"].astype(float).values; g=sub["study_uid"].values
    cnn=oof_v[idx,j]
    ex=[extra(to_display_uint8(*read_dicom(r["source_path"])) ) for _,r in sub.iterrows()]
    exdf=pd.DataFrame(ex)
    pool_base=["femur_margin_min_cm","femur_width_cm","femur_trochanter_bulge",
               "femur_shaft_deg","femur_axis_angle","femur_margin_top_cm",
               "femur_neck_width_mm","fatlas_resid_bone","fatlas_bone_iou","falign_angle"]
    pool=pool_base+list(exdf.columns)
    X=pd.concat([sub[pool_base].reset_index(drop=True), exdf],axis=1)
    TR=list(range(5)); TE=list(range(5,10))
    # одиночные на train
    res=[]
    for c in pool:
        a=cv(X[[c]].values,y,g,cnn,TR)
        res.append((a,c))
    res.sort(reverse=True)
    print("топ одиночных (train):")
    for a,c in res[:8]: print("  %-22s %.3f"%(c,a))
    # жадный отбор на train
    chosen=[]; best=-1
    while len(chosen)<5:
        gname,gscore=None,best
        for c in pool:
            if c in chosen: continue
            a=cv(X[chosen+[c]].values,y,g,cnn,TR)
            if a>gscore+0.004: gscore,gname=a,c
        if gname is None: break
        chosen.append(gname); best=gscore
        print("  + %-22s train=%.3f"%(gname,best))
    a_tr=cv(X[chosen].values,y,g,cnn,TR)
    a_te=cv(X[chosen].values,y,g,cnn,TE)
    # F1 на тесте
    f1s=[]
    for seed in TE:
        oof=np.full(len(y),np.nan)
        for tr,va in StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed).split(X[chosen].values,y,groups=g):
            if len(np.unique(y[tr]))<2: continue
            Xf=np.column_stack([cnn,X[chosen].values]); mm=clf().fit(Xf[tr],y[tr]); oof[va]=mm.predict_proba(Xf[va])[:,1]
        ok=np.isfinite(oof)
        f1s.append(f1_score(y[ok],(oof[ok]>=0.5).astype(int),zero_division=0))
    out=["ЛУЧШИЙ набор: %s"%chosen,
         "train AUC=%.3f  held-out AUC=%.3f  held-out F1=%.3f±%.3f"%(a_tr,a_te,np.mean(f1s),np.std(f1s))]
    txt="\n".join(out); open("femur_best.txt","w",encoding="utf-8").write(txt)
    print(txt.encode("ascii","replace").decode())

if __name__=="__main__":
    main()