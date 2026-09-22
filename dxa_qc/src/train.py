# -*- coding: utf-8 -*-
"""Обучение мультизадачной модели контроля качества DXA.

Разбиение — GroupKFold по study_uid (исключает утечку между train/val).
Сохранение — ансамбль из K фолдов + модель, обученная на всех данных.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from . import config as C
from . import dataset as ds
from . import model as model_lib


# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def masked_bce(logits, target, mask, pos_weight=None):
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=pos_weight)
    mask = mask.float()
    denom = mask.sum().clamp(min=1.0)
    return (loss * mask).sum() / denom


def compute_pos_weight(df: pd.DataFrame, column: str) -> torch.Tensor:
    vals = df[column].dropna().values
    if len(vals) == 0:
        return None
    pos = float((vals > 0.5).sum())
    neg = float(len(vals) - pos)
    if pos == 0:
        return None
    return torch.tensor([neg / pos], dtype=torch.float32)


def compute_pos_weight_by_region(df: pd.DataFrame, column: str) -> torch.Tensor:
    """pos_weight для головы качества: по одному значению на анатомическую область."""
    w = torch.ones(len(C.REGIONS), dtype=torch.float32)
    for i, region in enumerate(C.REGIONS):
        vals = df[df["region"] == region][column].dropna().values
        if len(vals) == 0:
            continue
        pos = float((vals > 0.5).sum())
        neg = float(len(vals) - pos)
        if pos > 0:
            w[i] = neg / pos
    return w


def compute_pos_weight_viol(df: pd.DataFrame) -> torch.Tensor:
    w = torch.ones(C.N_VIOLATIONS, dtype=torch.float32)
    for i, name in enumerate(C.VIOLATIONS):
        vals = df["viol_" + name].dropna().values
        if len(vals) == 0:
            continue
        pos = float((vals > 0.5).sum())
        neg = float(len(vals) - pos)
        if pos > 0:
            w[i] = neg / pos
    return w


# --------------------------------------------------------------------------- #
def train_epoch(model, loader, optimizer, device, pos_w_q, pos_w_v):
    model.train()
    total = 0.0
    for batch in loader:
        x = batch["image"].to(device)
        region = batch["region"].to(device)
        out = model(x)

        l_region = nn.functional.cross_entropy(out["region"], region)

        # выбираем логит качества, соответствующий области изображения
        q_logit = out["quality"].gather(1, region.unsqueeze(1)).squeeze(1)
        pw_q = None
        if pos_w_q is not None:
            pw_q = pos_w_q.to(device)[region]
        l_quality = masked_bce(
            q_logit, batch["quality"].to(device),
            batch["quality_mask"].to(device), pw_q)

        l_viol = masked_bce(
            out["violation"], batch["viol"].to(device),
            batch["viol_mask"].to(device),
            pos_w_v.to(device) if pos_w_v is not None else None)

        loss = 1.0 * l_region + 1.0 * l_quality + 1.0 * l_viol
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss) * len(x)
    return total / max(len(loader.dataset), 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs_q, probs_v, probs_r, idxs = [], [], [], []
    for batch in loader:
        x = batch["image"].to(device)
        out = model(x)
        probs_q.append(torch.sigmoid(out["quality"]).cpu().numpy())
        probs_v.append(torch.sigmoid(out["violation"]).cpu().numpy())
        probs_r.append(torch.softmax(out["region"], dim=1).cpu().numpy())
        idxs.append(batch["index"].numpy())
    return (np.concatenate(probs_q), np.concatenate(probs_v),
            np.concatenate(probs_r), np.concatenate(idxs))


# --------------------------------------------------------------------------- #
def bootstrap_ci(y, p, metric, n=1000, seed=0, thr=0.5):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    vals = []
    N = len(y)
    if N == 0:
        return (float("nan"), float("nan"))
    for _ in range(n):
        idx = rng.integers(0, N, N)
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2:
            continue
        try:
            if metric == "f1":
                vals.append(f1_score(yy, (pp >= thr).astype(int), zero_division=0))
            elif metric == "auc":
                vals.append(roc_auc_score(yy, pp))
            elif metric == "ap":
                vals.append(average_precision_score(yy, pp))
        except Exception:
            continue
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def best_threshold(y, p):
    ths = np.linspace(0.05, 0.95, 91)
    best, bt = -1.0, 0.5
    for t in ths:
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best:
            best, bt = f, float(t)
    return bt, best


def evaluate_quality(df, probs, mask_idx):
    y = df["quality"].values.astype(float)
    valid = ~np.isnan(y) & mask_idx
    y = y[valid]
    p = probs[valid]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return dict(n=int(len(y)), f1=None)
    thr, f1_t = best_threshold(y, p)
    res = dict(
        n=int(len(y)),
        pos=int((y > 0.5).sum()),
        auc=float(roc_auc_score(y, p)),
        auc_ci=bootstrap_ci(y, p, "auc"),
        ap=float(average_precision_score(y, p)),
        ap_ci=bootstrap_ci(y, p, "ap"),
        f1_at_05=float(f1_score(y, (p >= 0.5).astype(int), zero_division=0)),
        f1_best=float(f1_t),
        thr_best=thr,
        precision=float(precision_score(y, (p >= thr).astype(int), zero_division=0)),
        recall=float(recall_score(y, (p >= thr).astype(int), zero_division=0)),
        balanced_acc=float(balanced_accuracy_score(y, (p >= thr).astype(int))),
    )
    tn = int(((y < 0.5) & (p < thr)).sum())
    fp = int(((y < 0.5) & (p >= thr)).sum())
    fn = int(((y > 0.5) & (p < thr)).sum())
    tp = int(((y > 0.5) & (p >= thr)).sum())
    res.update(tp=tp, fp=fp, fn=fn, tn=tn)
    res["specificity"] = tn / max(tn + fp, 1)
    res["sensitivity"] = tp / max(tp + fn, 1)
    return res


# --------------------------------------------------------------------------- #
def run_cv(df: pd.DataFrame, args):
    device = args.device
    labeled = df[df["quality"].notna()].copy()

    gkf = GroupKFold(n_splits=args.folds)
    groups = df["study_uid"].values

    oof_q = np.full(len(df), np.nan)
    oof_v = np.full((len(df), C.N_VIOLATIONS), np.nan)
    oof_r = np.full((len(df), len(C.REGIONS)), np.nan)

    pos_w_q = compute_pos_weight_by_region(labeled, "quality")
    pos_w_v = compute_pos_weight_viol(labeled)

    for k, (tr_idx, va_idx) in enumerate(gkf.split(df, groups=groups)):
        tr = df.iloc[tr_idx].reset_index(drop=True)
        va = df.iloc[va_idx].reset_index(drop=True)
        tr_lab = tr[tr["quality"].notna()]

        set_seed(C.SEED + k)
        model = model_lib.build_model(pretrained=True).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=C.WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

        tr_loader = DataLoader(ds.DXADataset(tr_lab, train=True, size=args.size),
                               batch_size=args.batch, shuffle=True,
                               num_workers=C.NUM_WORKERS)
        va_loader = DataLoader(ds.DXADataset(va, train=False, size=args.size),
                               batch_size=args.batch, shuffle=False,
                               num_workers=C.NUM_WORKERS)

        # --- обучение с выбором лучшей эпохи по комбинированной метрике ---
        # score = mean(ROC-AUC качества, macro ROC-AUC типов нарушений)
        yv_all = va["quality"].values.astype(float)
        reg_all = va["region_idx"].values
        va_real = ds.real_mask(va)   # выбор эпохи — тоже по реальным снимкам
        viol_y = {name: va["viol_" + name].values.astype(float) for name in C.VIOLATIONS}
        best_score, best_state, best_ep = -1.0, None, -1
        for ep in range(args.epochs):
            tr_loss = train_epoch(model, tr_loader, opt, device, pos_w_q, pos_w_v)
            sched.step()

            # периодическая валидация (каждые 3 эпохи и последняя)
            if ep == args.epochs - 1 or (ep + 1) % 3 == 0:
                pq_v, pv_v, _pr, idx_v = predict(model, va_loader, device)
                reg_v = reg_all[idx_v]
                q_v = pq_v[np.arange(len(va)), reg_v]
                vv = ~np.isnan(yv_all) & va_real
                q_auc = -1.0
                if vv.sum() and len(np.unique(yv_all[vv])) > 1:
                    q_auc = float(roc_auc_score(yv_all[vv], q_v[vv]))
                # macro AUC по типам нарушений
                v_aucs = []
                for j, name in enumerate(C.VIOLATIONS):
                    yy = viol_y[name]
                    val = ~np.isnan(yy) & va_real
                    if val.sum() and len(np.unique(yy[val])) > 1:
                        v_aucs.append(roc_auc_score(yy[val], pv_v[val, j]))
                v_auc = float(np.mean(v_aucs)) if v_aucs else -1.0
                parts = [x for x in (q_auc, v_auc) if x >= 0]
                score = float(np.mean(parts)) if parts else -1.0
                if score > best_score:
                    best_score = score
                    best_ep = ep
                    best_state = {kk: vv2.detach().clone()
                                  for kk, vv2 in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)

        pq, pv, pr, idx = predict(model, va_loader, device)
        gidx = va_idx[idx]          # локальные индексы -> глобальные
        reg = va["region_idx"].values
        q_fold = pq[np.arange(len(va)), reg]        # вероятность качества своей области
        oof_q[gidx] = q_fold
        oof_v[gidx] = pv
        oof_r[gidx] = pr
        torch.save(model.state_dict(), C.FOLD_WEIGHTS_TMPL.format(k=k))

        # метрики текущего фолда (для мониторинга)
        yv = va["quality"].values.astype(float)
        val = ~np.isnan(yv)
        f1 = auc = float("nan")
        if val.sum() and len(np.unique(yv[val])) > 1:
            _, f1 = best_threshold(yv[val], q_fold[val])
            auc = float(roc_auc_score(yv[val], q_fold[val]))
        acc_r = float((pr.argmax(1) == reg).mean())
        print(f"  fold {k}: best_ep={best_ep}  train_loss={tr_loss:.4f}  val_n={len(va)}  "
              f"q_f1={f1:.3f}  q_auc={auc:.3f}  region_acc={acc_r:.3f}")

    # метрики считаем ТОЛЬКО по реальным снимкам: синтетика нужна для обучения
    real = ds.real_mask(df)
    metrics = {}
    metrics["quality"] = evaluate_quality(df, oof_q, real)
    region_valid = ~np.isnan(oof_r[:, 0]) & real
    metrics["region_accuracy"] = float(
        (oof_r.argmax(1) == df["region_idx"].values)[region_valid].mean()
        if np.any(region_valid) else float("nan"))

    # нарушения (мультилейбл, macro-F1 по критериям с метками)
    viol_metrics = {}
    f1s = []
    for i, name in enumerate(C.VIOLATIONS):
        col = "viol_" + name
        if col not in df:
            continue
        y = df[col].values.astype(float)
        valid = ~np.isnan(y) & ~np.isnan(oof_v[:, i]) & real
        if valid.sum() == 0 or len(np.unique(y[valid])) < 2:
            viol_metrics[name] = dict(n=int(valid.sum()), f1=None)
            continue
        yv, pv = y[valid], oof_v[valid, i]
        thr, f1t = best_threshold(yv, pv)
        viol_metrics[name] = dict(
            n=int(valid.sum()), pos=int((yv > 0.5).sum()),
            auc=float(roc_auc_score(yv, pv)),
            f1_best=float(f1t), thr_best=thr,
            f1_at_05=float(f1_score(yv, (pv >= 0.5).astype(int), zero_division=0)),
        )
        f1s.append(f1t)
    metrics["violations"] = viol_metrics
    metrics["violation_macro_f1"] = float(np.mean(f1s)) if f1s else None

    # сохранить OOF
    np.save(os.path.join(C.ARTIFACTS_DIR, "oof_quality.npy"), oof_q)
    np.save(os.path.join(C.ARTIFACTS_DIR, "oof_violation.npy"), oof_v)
    np.save(os.path.join(C.ARTIFACTS_DIR, "oof_region.npy"), oof_r)
    with open(C.METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=float)
    return metrics, oof_q, oof_v, oof_r


def train_full(df: pd.DataFrame, args):
    """Обучить финальную модель на всех размеченных данных."""
    device = args.device
    labeled = df[df["quality"].notna()].reset_index(drop=True)
    set_seed(C.SEED)
    model = model_lib.build_model(pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loader = DataLoader(ds.DXADataset(labeled, train=True, size=args.size),
                        batch_size=args.batch, shuffle=True,
                        num_workers=C.NUM_WORKERS)
    pos_w_q = compute_pos_weight_by_region(labeled, "quality")
    pos_w_v = compute_pos_weight_viol(labeled)
    for ep in range(args.epochs):
        loss = train_epoch(model, loader, opt, device, pos_w_q, pos_w_v)
        sched.step()
    torch.save(model.state_dict(), C.BEST_WEIGHTS)
    print(f"[full] сохранено: {C.BEST_WEIGHTS} (loss={loss:.4f})")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=C.DEFAULT_DATASET_ROOT)
    ap.add_argument("--manifest", default=C.MANIFEST_CSV)
    ap.add_argument("--rebuild-manifest", action="store_true")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--folds", type=int, default=C.N_FOLDS)
    ap.add_argument("--batch", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--size", type=int, default=C.IMAGE_SIZE)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    os.makedirs(C.ARTIFACTS_DIR, exist_ok=True)
    if args.rebuild_manifest or not os.path.isfile(args.manifest):
        df = ds.build_manifest(args.dataset_root, args.manifest)
    else:
        df = ds.load_manifest(args.manifest)
    print(f"[data] всего изображений: {len(df)}  устройство: {args.device}")

    t0 = time.time()
    metrics, oof_q, oof_v, oof_r = run_cv(df, args)
    print(f"\n[CV] время: {time.time()-t0:.1f}s")
    q = metrics["quality"]
    print("[CV] quality:", json.dumps(q, ensure_ascii=False, default=float))
    print("[CV] region_accuracy:", round(metrics["region_accuracy"], 4))
    print("[CV] violation_macro_f1:", metrics["violation_macro_f1"])
    for name, m in metrics["violations"].items():
        print(f"    {name}: {m}")

    train_full(df, args)


if __name__ == "__main__":
    main()
