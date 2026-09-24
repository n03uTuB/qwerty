# -*- coding: utf-8 -*-
"""Финальная проверка наборов femur_positioning (train сиды 0-4, held-out 5-9)."""
from __future__ import annotations
import os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src.data import load_manifest
from src.features import load_features

def clf():
    return Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(max_iter=1000, class_weight="balanced"))])
def ev(X,y,g,cnn,seeds,mode):
    A,F=[],[]
    for seed in seeds:
        oof=np.full(len(y),np.nan)
        for tr,va in StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed).split(X,y,groups=g):
            if len(np.unique(y[tr]))<2: continue
            if mode=="cnn": oof[va]=cnn[va]
            else:
                Xf=np.column_stack([cnn,X]) if mode=="fused" else X
                m=clf().fit(Xf[tr],y[tr]); oof[va]=m.predict_proba(Xf[va])[:,1]
        ok=np.isfinite(oof)
        if len(np.unique(y[ok]))<2: continue
        A.append(roc_auc_score(y[ok],oof[ok])); F.append(f1_score(y[ok],(oof[ok]>=0.5).astype(int),zero_division=0))
    return np.mean(A),np.std(A),np.mean(F),np.std(F)

def main():
    man=load_manifest(C.MANIFEST_CSV); df=load_features(man)
    oof_v=np.load(C.OOF_VIOLATION_NPY); j=C.VIOLATION_IDX["femur_positioning"]
    m=df["region"].isin([C.REGION_FEMUR_LEFT,C.REGION_FEMUR_RIGHT]).values & df["viol_femur_positioning"].notna().values
    sub=df[m].reset_index(drop=True); idx=df.index[m].values
    y=sub["viol_femur_positioning"].astype(float).values; g=sub["study_uid"].values
    cnn=oof_v[idx,j]
    TR=list(range(5)); TE=list(range(5,10))
    sets={
        "cur geo [margin_min,width]":(["femur_margin_min_cm","femur_width_cm"],"geo"),
        "width+axis+bboxh":(["femur_width_cm","femur_axis_angle","femur_bbox_h_ratio"],"fused"),
        "width+bboxh+headdx":(["femur_width_cm","femur_bbox_h_ratio","femur_head_dx"],"fused"),
        "width+bboxh":(["femur_width_cm","femur_bbox_h_ratio"],"fused"),
        "width+bboxh+bulge":(["femur_width_cm","femur_bbox_h_ratio","femur_trochanter_bulge"],"fused"),
        "width+margin+bboxh":(["femur_width_cm","femur_margin_min_cm","femur_bbox_h_ratio"],"fused"),
        "width+bboxh+headdx+bulge":(["femur_width_cm","femur_bbox_h_ratio","femur_head_dx","femur_trochanter_bulge"],"fused"),
    }
    out=[]
    for name,(cols,mode) in sets.items():
        X=sub[cols].astype(float).fillna(0).values
        atr,_,_,_=ev(X,y,g,cnn,TR,mode)
        ate,astd,fm,fs=ev(X,y,g,cnn,TE,mode)
        out.append((ate,astd,fm,name,mode,atr))
    out.sort(reverse=True)
    txt=["=== femur_positioning: train/held-out ==="]
    for ate,astd,fm,name,mode,atr in out:
        txt.append("  %-28s [%s] train=%.3f held=%.3f±%.3f F1=%.3f"%(name,mode,atr,ate,astd,fm))
    t="\n".join(txt); open("femur_final.txt","w",encoding="utf-8").write(t)
    print(t.encode("ascii","replace").decode())

if __name__=="__main__":
    main()