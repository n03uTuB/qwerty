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
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedGroupKFold
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

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", enabled=use_amp):
            out = model(x)
            loss, _parts = model_lib.compute_loss(
                out, region, quality, qmask, viol, vmask, pos_w_q, pos_w_v,
                mode=loss_mode)
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
        q, v, r = model_lib.predict_probs(model, x, tta=tta)
        probs_q.append(q.cpu().numpy())
        probs_v.append(v.cpu().numpy())
        probs_r.append(r.cpu().numpy())
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


def prior_threshold(y, p):
    """Порог по ожидаемой доле нарушений.

    Нарушением объявляем столько снимков, сколько их ожидается по обучающей выборке.
    При 6–36 положительных примерах порог, максимизирующий F1, скачет от фолда к
    фолду, а доля нарушений — самая надёжная величина, которая у нас есть.
    """
    prevalence = float(np.mean(y)) if len(y) else 0.0
    if prevalence <= 0 or prevalence >= 1 or len(p) == 0:
        return 0.5
    count = max(1, int(round(prevalence * len(p))))
    return float(np.sort(p)[::-1][min(count, len(p)) - 1])


def pick_threshold(y, p, mode: str = None):
    """Выбрать порог и вернуть (порог, F1). Режим — из config.THRESHOLD_MODE."""
    mode = mode or getattr(C, "THRESHOLD_MODE", "prior")
    if mode == "prior":
        thr = prior_threshold(y, p)
        return thr, float(f1_score(y, (p >= thr).astype(int), zero_division=0))
    return best_threshold(y, p)


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
def _splitter(args, df, groups):
    """Сплиттер с группировкой по study_uid.

    StratifiedGroupKFold дополнительно выравнивает состав анатомических областей
    между фолдами (стратификация по region_idx) — это важно, т.к. позвоночник и
    бедро имеют разные критерии и разный баланс классов.
    """
    if C.USE_STRATIFIED_GROUP:
        return StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                                    random_state=C.SEED)
    return GroupKFold(n_splits=args.folds)


def _inner_score(model, loader, device) -> float:
    """Комбинированная метрика для выбора эпохи на внутреннем холдауте.

    score = среднее из ROC-AUC качества и macro ROC-AUC типов нарушений
    (считается только по реальным снимкам, как и отчётные метрики).
    """
    pq, pv, _pr, idx = predict(model, loader, device, tta=False)
    df = loader.dataset.df
    real = ds.real_mask(df)
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

    splitter = _splitter(args, df, df["study_uid"].values)
    groups = df["study_uid"].values
    y_strat = df["region_idx"].values

    oof_q = np.full(len(df), np.nan)
    oof_v = np.full((len(df), C.N_VIOLATIONS), np.nan)
    oof_r = np.full((len(df), len(C.REGIONS)), np.nan)

    pos_w_q = compute_pos_weight_by_region(labeled, "quality")
    pos_w_v = compute_pos_weight_viol(labeled)
    tta = bool(getattr(args, "tta", C.USE_TTA))
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and C.USE_AMP))

    for k, (tr_idx, va_idx) in enumerate(splitter.split(df, y_strat, groups=groups)):
        tr = df.iloc[tr_idx].reset_index(drop=True)
        va = df.iloc[va_idx].reset_index(drop=True)
        tr_lab = tr[tr["quality"].notna()]

        set_seed(C.SEED + k)
        model = model_lib.build_model(pretrained=True).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=C.WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

        # --- честный выбор лучшей эпохи: внутренний холдаут ВНУТРИ обучающего фолда ---
        # Раньше эпоха выбиралась по валидационному фолду, из которого затем брались
        # OOF-предсказания -> утечка и завышение метрик. Теперь выбор идёт по
        # отдельным исследованиям, которые не участвуют в OOF.
        inner_tr = tr_lab
        inner_va = None
        if getattr(args, "select_epoch", True) and len(tr_lab) > 20:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=C.SEED + k)
            a, b = next(gss.split(tr_lab, groups=tr_lab["study_uid"].values))
            inner_tr = tr_lab.iloc[a].reset_index(drop=True)
            inner_va = tr_lab.iloc[b].reset_index(drop=True)

        tr_loader = DataLoader(ds.DXADataset(inner_tr, train=True, size=args.size),
                               batch_size=args.batch, shuffle=True,
                               num_workers=C.NUM_WORKERS)
        inner_loader = (DataLoader(ds.DXADataset(inner_va, train=False, size=args.size),
                                   batch_size=args.batch, shuffle=False,
                                   num_workers=C.NUM_WORKERS)
                        if inner_va is not None else None)
        va_loader = DataLoader(ds.DXADataset(va, train=False, size=args.size),
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
    sampler = ds.make_balanced_sampler(labeled) if C.BALANCED_SAMPLER else None
    loader = DataLoader(ds.DXADataset(labeled, train=True, size=args.size),
                        batch_size=args.batch, shuffle=sampler is None,
                        sampler=sampler, num_workers=C.NUM_WORKERS)
    pos_w_q = compute_pos_weight_by_region(labeled, "quality")
    pos_w_v = compute_pos_weight_viol(labeled)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and C.USE_AMP))
    for ep in range(args.epochs):
        loss = train_epoch(model, loader, opt, device, pos_w_q, pos_w_v,
                           scaler=scaler, loss_mode=C.QUALITY_LOSS)
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
    ap.add_argument("--backbone", default=C.BACKBONE)
    ap.add_argument("--no-tta", action="store_true", help="отключить TTA при оценке")
    ap.add_argument("--no-select-epoch", action="store_true",
                    help="не выбирать эпоху (обучить фиксированное число эпох)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.tta = not args.no_tta
    args.select_epoch = not args.no_select_epoch

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    os.makedirs(C.ARTIFACTS_DIR, exist_ok=True)
    if args.backbone != C.BACKBONE:
        C.BACKBONE = args.backbone
    if args.rebuild_manifest or not os.path.isfile(args.manifest):
        df = ds.build_manifest(args.dataset_root, args.manifest)
    else:
        df = ds.load_manifest(args.manifest)
    print(f"[data] всего изображений: {len(df)}  устройство: {args.device}  "
          f"бэкбон: {C.BACKBONE}  TTA: {args.tta}")

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
