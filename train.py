# train.py
from __future__ import annotations
import os
import math
import json
import argparse
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Any

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Subset

from data import RadiationDataset, collate_batch, FeatureStandardizer, scaffold_for_row
from models.predictor import build_from_sample, DoseConstantPredictor


# =============================
# Utils
# =============================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(pack, device: torch.device):
    pack.atom_batch = pack.atom_batch.to(device)
    pack.frag_batch = pack.frag_batch.to(device)
    pack.num_feats = pack.num_feats.to(device)
    if pack.y is not None:
        pack.y = pack.y.to(device)
    return pack


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = y_true
    yp = y_pred
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def scaffold_groups(smiles_list: List[str]) -> Dict[str, List[int]]:
    """
    Группирует индексы по scaffold SMILES.
    """
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        scaf = scaffold_for_row(smi)
        groups[scaf].append(i)
    return groups


def scaffold_kfold_indices(smiles: List[str], n_splits: int = 5, seed: int = 42) -> List[Tuple[List[int], List[int]]]:
    """
    Возвращает список (train_idx, val_idx) для scaffold k-fold.
    Стратегия: сортируем scaffold‑группы по убыванию размера и по очереди кладём в фолды (greedy).
    """
    rng = random.Random(seed)
    groups = scaffold_groups(smiles)
    scafs = list(groups.items())
    scafs.sort(key=lambda kv: len(kv[1]), reverse=True)

    folds: List[List[int]] = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits

    for scaf, idxs in scafs:
        # кладём группу в фолд с текущим минимальным размером
        k = int(np.argmin(fold_sizes))
        folds[k].extend(idxs)
        fold_sizes[k] += len(idxs)

    splits: List[Tuple[List[int], List[int]]] = []
    all_idx = set(range(len(smiles)))
    for k in range(n_splits):
        val_idx = sorted(folds[k])
        train_idx = sorted(list(all_idx - set(val_idx)))
        splits.append((train_idx, val_idx))
    return splits


def build_model_from_loader(loader: DataLoader, d_model: int, gnn_layers: int, dropout: float, device: torch.device) -> DoseConstantPredictor:
    first_pack = next(iter(loader))
    sample = {
        "atom_batch": first_pack.atom_batch,
        "frag_batch": first_pack.frag_batch,
        "num_feats": first_pack.num_feats,
    }
    model = build_from_sample(
        sample=sample, d_model=d_model, gnn_layers=gnn_layers, dropout=dropout, use_mamba_in_fuser=True
    ).to(device)
    return model


def try_load_ssl_checkpoint(model: DoseConstantPredictor, standardizer: FeatureStandardizer, ckpt_path: str):
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print(f"[i] SSL checkpoint not provided or not found: {ckpt_path}")
        return
    payload = torch.load(ckpt_path, map_location="cpu")
    state = payload.get("model_state_dict", {})
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[✓] Loaded SSL weights from {ckpt_path}")
    if missing:
        print(f"[!] Missing keys: {len(missing)} (showing first 10): {missing[:10]}")
    if unexpected:
        print(f"[!] Unexpected keys: {len(unexpected)} (showing first 10): {unexpected[:10]}")

    mean_np = payload.get("standardizer_mean", None)
    std_np = payload.get("standardizer_std", None)
    if mean_np is not None and std_np is not None:
        standardizer.mean_ = torch.tensor(mean_np, dtype=torch.float)
        standardizer.std_ = torch.tensor(std_np, dtype=torch.float)
        print("[✓] Loaded feature standardizer stats from SSL checkpoint.")


def evaluate(model: DoseConstantPredictor, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    y_true_all: List[float] = []
    y_pred_all: List[float] = []
    with torch.no_grad():
        for pack in loader:
            pack = to_device(pack, device)
            out = model(
                atom_batch=pack.atom_batch,
                frag_batch=pack.frag_batch,
                num_feats=pack.num_feats,
                ssl_mask_cfg=None,
                return_ssl=False,
            )
            pred = out["pred"].squeeze(-1).detach().cpu().numpy()
            y = pack.y.squeeze(-1).detach().cpu().numpy()
            y_true_all.append(y)
            y_pred_all.append(pred)
    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
    }


# =============================
# Training loop (per-fold)
# =============================
def train_one_fold(
    ds: RadiationDataset,
    train_idx: List[int],
    val_idx: List[int],
    args: argparse.Namespace,
    device: torch.device,
    fold_id: int,
) -> Dict[str, Any]:
    # Datasets
    ds_train = Subset(ds, train_idx)
    ds_val = Subset(ds, val_idx)

    # Standardizer fit on train only
    stdzr = FeatureStandardizer().fit(ds, indices=train_idx)

    def collate_and_standardize(items):
        pack = collate_batch(items)
        pack.num_feats = stdzr.transform(pack.num_feats)
        return pack

    train_loader = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate_and_standardize, drop_last=False
    )
    val_loader = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_and_standardize, drop_last=False
    )

    # Model
    model = build_model_from_loader(
        loader=train_loader, d_model=args.d_model, gnn_layers=args.gnn_layers, dropout=args.dropout, device=device
    )

    # Optionally load SSL checkpoint
    if args.ssl_ckpt:
        try_load_ssl_checkpoint(model, stdzr, args.ssl_ckpt)

    # Optimizer & scheduler
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, args.epochs))
    else:
        scheduler = None

    best_val_rmse = float("inf")
    best_state = None
    epochs_no_improve = 0

    # Train
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for pack in train_loader:
            pack = to_device(pack, device)

            out = model(
                atom_batch=pack.atom_batch,
                frag_batch=pack.frag_batch,
                num_feats=pack.num_feats,
                ssl_mask_cfg=None,
                return_ssl=False,
            )
            pred = out["pred"]  # [B,1]
            loss = nn.functional.mse_loss(pred, pack.y)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()

            total_loss += float(loss.item())
            n_batches += 1

        if scheduler is not None:
            scheduler.step()

        train_loss = total_loss / max(1, n_batches)

        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        val_rmse = val_metrics["RMSE"]

        print(f"[Fold {fold_id}] Epoch {epoch:03d} | train_MSE: {train_loss:.6f} | val_RMSE: {val_rmse:.6f} | val_MAE: {val_metrics['MAE']:.6f} | val_R2: {val_metrics['R2']:.4f}")

        # Early stopping
        if val_rmse + 1e-9 < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {
                "model": model.state_dict(),
                "std_mean": stdzr.mean_.cpu().numpy(),
                "std_std": stdzr.std_.cpu().numpy(),
                "epoch": epoch,
                "val_metrics": {k: float(v) for k, v in val_metrics.items() if isinstance(v, (int, float))},
                "args": vars(args),
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"[Fold {fold_id}] Early stopping at epoch {epoch} (no improvement {args.patience} epochs).")
                break

    # Reload best & final eval (both train and val)
    if best_state is not None:
        model.load_state_dict(best_state["model"])

    # Build final loaders again to ensure same standardization
    train_eval_loader = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_and_standardize, drop_last=False
    )
    val_eval_loader = val_loader

    train_metrics = evaluate(model, train_eval_loader, device)
    val_metrics = evaluate(model, val_eval_loader, device)

    # Save best
    os.makedirs(args.out_dir, exist_ok=True)
    fold_path = os.path.join(args.out_dir, f"fold{fold_id}_best.pt")
    torch.save(best_state if best_state is not None else {"model": model.state_dict()}, fold_path)
    print(f"[✓] Saved best fold {fold_id} checkpoint to {fold_path}")

    # Save json metrics
    report = {
        "fold": fold_id,
        "best_epoch": int(best_state["epoch"]) if best_state and "epoch" in best_state else None,
        "train": {k: float(v) for k, v in train_metrics.items() if isinstance(v, (int, float))},
        "val": {k: float(v) for k, v in val_metrics.items() if isinstance(v, (int, float))},
    }
    with open(os.path.join(args.out_dir, f"fold{fold_id}_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[✓] Saved fold {fold_id} metrics json.")

    return {
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "fold_ckpt": fold_path,
    }


# =============================
# Main
# =============================
def main():
    parser = argparse.ArgumentParser(description="Supervised training with scaffold k-fold CV for dose constant regression")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV (smiles, solv_smiles, diel_const, conc_stand, dc_stand)")
    parser.add_argument("--ssl_ckpt", type=str, default="", help="Optional path to SSL checkpoint (ssl_pretrained.pt)")
    parser.add_argument("--out_dir", type=str, default="runs")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine"])
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--gnn_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    ds = RadiationDataset(args.csv)
    smiles = ds.df["smiles"].astype(str).tolist()

    # Scaffold k-fold splits
    folds = scaffold_kfold_indices(smiles, n_splits=args.folds, seed=args.seed)
    print("[i] Fold sizes:", [len(v) for (_, v) in folds])

    # Run folds
    all_val_rmse = []
    all_val_mae = []
    all_val_r2 = []

    for k, (train_idx, val_idx) in enumerate(folds, start=1):
        print(f"\n========== Fold {k}/{args.folds} ==========")
        result = train_one_fold(ds, train_idx, val_idx, args, device, fold_id=k)

        vm = result["val_metrics"]
        all_val_rmse.append(vm["RMSE"])
        all_val_mae.append(vm["MAE"])
        all_val_r2.append(vm["R2"])

    # Summary
    def mean_std(arr):
        return float(np.mean(arr)), float(np.std(arr))

    rmse_mean, rmse_std = mean_std(all_val_rmse)
    mae_mean, mae_std = mean_std(all_val_mae)
    r2_mean, r2_std = mean_std(all_val_r2)

    print("\n===== CV Summary =====")
    print(f"RMSE: {rmse_mean:.6f} ± {rmse_std:.6f}")
    print(f"MAE : {mae_mean:.6f} ± {mae_std:.6f}")
    print(f"R2  : {r2_mean:.4f} ± {r2_std:.4f}")

    # Save summary
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "cv_summary.json"), "w") as f:
        json.dump({
            "folds": args.folds,
            "RMSE_mean": rmse_mean, "RMSE_std": rmse_std,
            "MAE_mean": mae_mean, "MAE_std": mae_std,
            "R2_mean": r2_mean, "R2_std": r2_std,
            "val_rmse_per_fold": all_val_rmse,
            "val_mae_per_fold": all_val_mae,
            "val_r2_per_fold": all_val_r2,
            "args": vars(args),
        }, f, indent=2)
    print(f"[✓] Saved CV summary to {os.path.join(args.out_dir, 'cv_summary.json')}")

    print("\nTip: для совсем маленького датасета (44 образца) полезно попробовать folds=11 (LOO по scaffold‑группам) "
          "и/или уменьшить d_model, повысить dropout, включить раннюю остановку построже.")


if __name__ == "__main__":
    main()