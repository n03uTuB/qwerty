# -*- coding: utf-8 -*-
"""Обучение мультизадачной модели контроля качества DXA.

Честный протокол:
  * разбиение StratifiedGroupKFold по study_uid (пациент не пересекает фолды);
  * лучшая эпоха выбирается на ВНУТРЕННЕМ холдауте внутри обучающего фолда
    (а не по валидационному, из которого затем берутся OOF — это была утечка);
  * пороги для метрики организатора выбираются ТОЛЬКО по train-части фолда,
    затем применяются к val-части и пулятся (как у организатора).

Артефакты: OOF-вероятности (quality/violation/region), fold_id, train-часть
вероятностей (для порогов) и metrics.json. Финальная модель обучается на всех
размеченных данных -> artifacts/best_model.pt.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit,
                                     StratifiedGroupKFold)
from torch.utils.data import DataLoader

from . import config as C
from .data import DXADataset, build_manifest, load_manifest, make_balanced_sampler, real_mask
from .metrics import (ORG_LABEL_NAMES, bootstrap_ci, bootstrap_ci_by_study,
                      organizer_macro_f1, org_label_truth, org_label_vector,
                      pick_threshold)
from .models import build_model, compute_loss, predict_probs

FOLD_ID_NPY = os.path.join(C.ARTIFACTS_DIR, "fold_id.npy")
TRAIN_QUALITY_NPY = os.path.join(C.ARTIFACTS_DIR, "train_quality.npy")
TRAIN_VIOLATION_NPY = os.path.join(C.ARTIFACTS_DIR, "train_violation.npy")


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
def train_epoch(model, loader, optimizer, device, pos_w_q, pos_w_v,
                scaler=None, loss_mode=C.QUALITY_LOSS):
    model.train()
    total = 0.0
    use_amp = scaler is not None and device == "cuda"
    for batch in loader:
        x = batch["image"].to(device)
        region = batch["region"].to(device)
        quality = batch["quality"].to(device)
        qmask = batch["quality_mask"].to(device)
        viol = batch["viol"].to(device)
        vmask = batch["viol_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            out = model(x)
            loss, _parts = compute_loss(out, region, quality, qmask, viol, vmask,
                                        pos_w_q, pos_w_v, mode=loss_mode)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += float(loss) * len(x)
    return total / max(len(loader.dataset), 1)


@torch.no_grad()
def predict(model, loader, device, tta: bool = False):
    model.eval()
    probs_q, probs_v, probs_r, idxs = [], [], [], []
    for batch in loader:
        x = batch["image"].to(device)
        q, v, r = predict_probs(model, x, tta=tta)
        probs_q.append(q.cpu().numpy())
        probs_v.append(v.cpu().numpy())
        probs_r.append(r.cpu().numpy())
        idxs.append(batch["index"].numpy())
    return (np.concatenate(probs_q), np.concatenate(probs_v),
            np.concatenate(probs_r), np.concatenate(idxs))


def evaluate_quality(df, probs, mask_idx) -> dict:
    y = df["quality"].values.astype(float)
    valid = ~np.isnan(y) & mask_idx
    y, p = y[valid], probs[valid]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return dict(n=int(len(y)), f1=None)
    thr, f1_t = pick_threshold(y, p)
    pred = (p >= thr).astype(int)
    res = dict(
        n=int(len(y)), pos=int((y > 0.5).sum()),
        auc=float(roc_auc_score(y, p)),
        auc_ci=bootstrap_ci(y, p, "auc"),
        ap=float(average_precision_score(y, p)),
        ap_ci=bootstrap_ci(y, p, "ap"),
        f1_at_05=float(f1_score(y, (p >= 0.5).astype(int), zero_division=0)),
        f1_best=float(f1_t), thr_best=float(thr),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        balanced_acc=float(balanced_accuracy_score(y, pred)),
    )
    tn = int(((y < 0.5) & (p < thr)).sum())
    fp = int(((y < 0.5) & (p >= thr)).sum())
    fn = int(((y > 0.5) & (p < thr)).sum())
    tp = int(((y > 0.5) & (p >= thr)).sum())
    res.update(tp=tp, fp=fp, fn=fn, tn=tn)
    res["specificity"] = tn / max(tn + fp, 1)
    res["sensitivity"] = tp / max(tp + fn, 1)
    return res


def _splitter(args):
    if C.USE_STRATIFIED_GROUP:
        return StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                                    random_state=C.SEED)
    return GroupKFold(n_splits=args.folds)


def _inner_score(model, loader, device) -> float:
    """Комбинированная метрика выбора эпохи (AUC качества + macro-AUC критериев)."""
    pq, pv, _pr, idx = predict(model, loader, device, tta=False)
    df = loader.dataset.df
    real = real_mask(df)
    yq = df["quality"].values.astype(float)[idx]
    reg = df["region_idx"].values[idx]
    q = pq[np.arange(len(idx)), reg]
    vv = ~np.isnan(yq) & real[idx]
    parts = []
    if vv.sum() and len(np.unique(yq[vv])) > 1:
        parts.append(float(roc_auc_score(yq[vv], q[vv])))
    v_aucs = []
    for j, name in enumerate(C.VIOLATIONS):
        y = df["viol_" + name].values.astype(float)[idx]
        ok = ~np.isnan(y) & real[idx]
        if ok.sum() and len(np.unique(y[ok])) > 1:
            v_aucs.append(roc_auc_score(y[ok], pv[ok, j]))
    if v_aucs:
        parts.append(float(np.mean(v_aucs)))
    return float(np.mean(parts)) if parts else -1.0


def run_cv(df: pd.DataFrame, args):
    device = args.device
    labeled = df[df["quality"].notna()].copy()
    splitter = _splitter(args)
    groups = df["study_uid"].values
    y_strat = df["region_idx"].values

    oof_q = np.full(len(df), np.nan)
    oof_v = np.full((len(df), C.N_VIOLATIONS), np.nan)
    oof_r = np.full((len(df), len(C.REGIONS)), np.nan)
    tr_q = np.full(len(df), np.nan)
    tr_v = np.full((len(df), C.N_VIOLATIONS), np.nan)
    fold_id = np.full(len(df), -1, dtype=int)

    pos_w_q = compute_pos_weight_by_region(labeled, "quality")
    pos_w_v = compute_pos_weight_viol(labeled)
    tta = bool(getattr(args, "tta", C.USE_TTA))
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and C.USE_AMP))

    for k, (tr_idx, va_idx) in enumerate(splitter.split(df, y_strat, groups=groups)):
        tr = df.iloc[tr_idx].reset_index(drop=True)
        va = df.iloc[va_idx].reset_index(drop=True)
        tr_lab = tr[tr["quality"].notna()]

        set_seed(C.SEED + k)
        model = build_model(pretrained=True).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=C.WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

        inner_tr, inner_va = tr_lab, None
        if getattr(args, "select_epoch", True) and len(tr_lab) > 20:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=C.SEED + k)
            a, b = next(gss.split(tr_lab, groups=tr_lab["study_uid"].values))
            inner_tr = tr_lab.iloc[a].reset_index(drop=True)
            inner_va = tr_lab.iloc[b].reset_index(drop=True)

        tr_loader = DataLoader(DXADataset(inner_tr, train=True, size=args.size),
                               batch_size=args.batch, shuffle=True,
                               num_workers=C.NUM_WORKERS)
        inner_loader = (DataLoader(DXADataset(inner_va, train=False, size=args.size),
                                   batch_size=args.batch, shuffle=False,
                                   num_workers=C.NUM_WORKERS)
                        if inner_va is not None else None)
        va_loader = DataLoader(DXADataset(va, train=False, size=args.size),
                               batch_size=args.batch, shuffle=False,
                               num_workers=C.NUM_WORKERS)
        tr_eval_loader = DataLoader(DXADataset(tr_lab, train=False, size=args.size),
                                    batch_size=args.batch, shuffle=False,
                                    num_workers=C.NUM_WORKERS)

        best_score, best_state, best_ep = -1.0, None, -1
        for ep in range(args.epochs):
            tr_loss = train_epoch(model, tr_loader, opt, device, pos_w_q, pos_w_v,
                                  scaler=scaler, loss_mode=C.QUALITY_LOSS)
            sched.step()
            if inner_loader is None:
                best_ep = ep
                continue
            if ep == args.epochs - 1 or (ep + 1) % 3 == 0:
                score = _inner_score(model, inner_loader, device)
                if score > best_score:
                    best_score, best_ep = score, ep
                    best_state = {kk: vv.detach().clone()
                                  for kk, vv in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)

        pq, pv, pr, idx = predict(model, va_loader, device, tta=tta)
        gidx = va_idx[idx]
        reg = va["region_idx"].values
        oof_q[gidx] = pq[np.arange(len(va)), reg]
        oof_v[gidx] = pv
        oof_r[gidx] = pr
        fold_id[gidx] = k

        # предсказания на обучающей части фолда — для честного выбора порога
        tq, tv, _tr_r, tidx = predict(model, tr_eval_loader, device, tta=tta)
        gtidx = tr_idx[tidx]
        treg = tr_lab["region_idx"].values
        tr_q[gtidx] = tq[np.arange(len(tr_lab)), treg]
        tr_v[gtidx] = tv

        torch.save(model.state_dict(), C.FOLD_WEIGHTS_TMPL.format(k=k))

        yv = va["quality"].values.astype(float)
        val = ~np.isnan(yv)
        f1 = auc = float("nan")
        if val.sum() and len(np.unique(yv[val])) > 1:
            _t, f1 = pick_threshold(yv[val], oof_q[gidx][val])
            auc = float(roc_auc_score(yv[val], oof_q[gidx][val]))
        acc_r = float((pr.argmax(1) == reg).mean())
        print(f"  fold {k}: best_ep={best_ep}  train_loss={tr_loss:.4f}  "
              f"val_n={len(va)}  q_f1={f1:.3f}  q_auc={auc:.3f}  region_acc={acc_r:.3f}")

    real = real_mask(df)
    metrics = {}
    metrics["quality"] = evaluate_quality(df, oof_q, real)
    region_valid = ~np.isnan(oof_r[:, 0]) & real
    metrics["region_accuracy"] = float(
        (oof_r.argmax(1) == df["region_idx"].values)[region_valid].mean()
        if np.any(region_valid) else float("nan"))

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
        thr, f1t = pick_threshold(yv, pv)
        viol_metrics[name] = dict(
            n=int(valid.sum()), pos=int((yv > 0.5).sum()),
            auc=float(roc_auc_score(yv, pv)),
            auc_ci=bootstrap_ci(yv, pv, "auc"),
            f1_best=float(f1t), thr_best=float(thr),
            f1_at_05=float(f1_score(yv, (pv >= 0.5).astype(int), zero_division=0)),
        )
        f1s.append(f1t)
    metrics["violations"] = viol_metrics
    metrics["violation_macro_f1"] = float(np.mean(f1s)) if f1s else None

    metrics["organizer"] = _organizer_honest(df, oof_v, tr_v, fold_id, real)

    os.makedirs(C.ARTIFACTS_DIR, exist_ok=True)
    np.save(C.OOF_QUALITY_NPY, oof_q)
    np.save(C.OOF_VIOLATION_NPY, oof_v)
    np.save(C.OOF_REGION_NPY, oof_r)
    np.save(FOLD_ID_NPY, fold_id)
    np.save(TRAIN_QUALITY_NPY, tr_q)
    np.save(TRAIN_VIOLATION_NPY, tr_v)
    with open(C.METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=float)
    return metrics, oof_q, oof_v, oof_r


def _organizer_honest(df, oof_v, tr_v, fold_id, real) -> dict:
    """macro-F1 организатора с порогами, выбранными ТОЛЬКО по train-части фолда."""
    truth = {name: org_label_truth(df, name) for name in ORG_LABEL_NAMES}
    pooled_pred = {name: np.full(len(df), np.nan) for name in ORG_LABEL_NAMES}
    folds = sorted(set(fold_id[fold_id >= 0].tolist()))
    for k in folds:
        va = np.where((fold_id == k) & real)[0]
        tr = np.where((fold_id != k) & real & ~np.isnan(tr_v[:, 0]))[0]
        if len(va) == 0:
            continue
        # порог по каждому критерию — на train-части, применяем к val-части
        crit_pred = {}
        for i, cname in enumerate(C.VIOLATIONS):
            y_tr = df["viol_" + cname].values.astype(float)[tr]
            p_tr = tr_v[tr, i]
            ok = ~np.isnan(y_tr) & ~np.isnan(p_tr)
            if ok.sum() and len(np.unique(y_tr[ok])) > 1:
                thr, _ = pick_threshold(y_tr[ok], p_tr[ok])
            else:
                thr = 0.5
            crit_pred[cname] = (oof_v[va, i] >= thr).astype(float)
        for name in ORG_LABEL_NAMES:
            members = [crit_pred[c] for c in C.ORG_LABELS[name]]
            pooled_pred[name][va] = np.max(np.vstack(members), axis=0)
    truths, preds = {}, {}
    for name in ORG_LABEL_NAMES:
        m = np.isfinite(pooled_pred[name])
        truths[name] = truth[name][m]
        preds[name] = pooled_pred[name][m]
    macro, per = organizer_macro_f1(truths, preds)
    return dict(macro=float(macro), per_label={k: float(v) for k, v in per.items()})


def train_full(df: pd.DataFrame, args):
    """Обучить финальную модель на всех размеченных данных."""
    device = args.device
    labeled = df[df["quality"].notna()].reset_index(drop=True)
    set_seed(C.SEED)
    model = build_model(pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    sampler = make_balanced_sampler(labeled) if C.BALANCED_SAMPLER else None
    loader = DataLoader(DXADataset(labeled, train=True, size=args.size),
                        batch_size=args.batch, shuffle=sampler is None,
                        sampler=sampler, num_workers=C.NUM_WORKERS)
    pos_w_q = compute_pos_weight_by_region(labeled, "quality")
    pos_w_v = compute_pos_weight_viol(labeled)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and C.USE_AMP))
    loss = 0.0
    for _ep in range(args.epochs):
        loss = train_epoch(model, loader, opt, device, pos_w_q, pos_w_v,
                           scaler=scaler, loss_mode=C.QUALITY_LOSS)
        sched.step()
    torch.save(model.state_dict(), C.BEST_WEIGHTS)
    print(f"[full] сохранено: {C.BEST_WEIGHTS} (loss={loss:.4f})")
    return model


def main():
    ap = argparse.ArgumentParser(description="Обучение гибридной DXA-QC модели")
    ap.add_argument("--dataset-root", default=C.DEFAULT_DATASET_ROOT)
    ap.add_argument("--manifest", default=C.MANIFEST_CSV)
    ap.add_argument("--rebuild-manifest", action="store_true")
    ap.add_argument("--add-synthetic", action="store_true",
                    help="добавить синтетические артефакты (только в обучение)")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--folds", type=int, default=C.N_FOLDS)
    ap.add_argument("--batch", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--size", type=int, default=C.IMAGE_SIZE)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--backbone", default=C.BACKBONE)
    ap.add_argument("--no-tta", action="store_true", help="отключить TTA при оценке")
    ap.add_argument("--no-select-epoch", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.tta = not args.no_tta
    args.select_epoch = not args.no_select_epoch

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    os.makedirs(C.ARTIFACTS_DIR, exist_ok=True)
    if args.backbone != C.BACKBONE:
        C.BACKBONE = args.backbone
    if args.rebuild_manifest or not os.path.isfile(args.manifest):
        df = build_manifest(args.dataset_root, args.manifest)
    else:
        df = load_manifest(args.manifest)
    if args.add_synthetic:
        from .data import add_synthetic
        before = len(df)
        df = add_synthetic(df)
        print(f"[data] добавлено синтетики: {len(df) - before} (только обучение)")
    print(f"[data] всего изображений: {len(df)}  устройство: {args.device}  "
          f"бэкбон: {C.BACKBONE}  TTA: {args.tta}  порог: {C.THRESHOLD_MODE}")

    t0 = time.time()
    metrics, _oq, _ov, _or = run_cv(df, args)
    print(f"\n[CV] время: {time.time()-t0:.1f}s")
    print("[CV] quality:", json.dumps(metrics["quality"], ensure_ascii=False, default=float))
    print("[CV] region_accuracy:", round(metrics["region_accuracy"], 4))
    print("[CV] violation_macro_f1:", metrics["violation_macro_f1"])
    print("[CV] organizer (честно):", json.dumps(metrics["organizer"],
                                                ensure_ascii=False, default=float))
    for name, m in metrics["violations"].items():
        print(f"    {name}: {m}")

    train_full(df, args)


if __name__ == "__main__":
    main()