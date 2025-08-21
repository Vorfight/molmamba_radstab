# predict.py
from __future__ import annotations
import os
import re
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from data import RadiationDataset, collate_batch, FeatureStandardizer
from models.predictor import build_from_sample, DoseConstantPredictor


def to_device(pack, device: torch.device):
    pack.atom_batch = pack.atom_batch.to(device)
    pack.frag_batch = pack.frag_batch.to(device)
    if hasattr(pack, "num_feats_mol") and pack.num_feats_mol is not None:
        pack.num_feats_mol = pack.num_feats_mol.to(device)
    if hasattr(pack, "num_feats_solv") and pack.num_feats_solv is not None:
        pack.num_feats_solv = pack.num_feats_solv.to(device)
    if hasattr(pack, "conc") and pack.conc is not None:
        pack.conc = pack.conc.to(device)
    if pack.y is not None:
        pack.y = pack.y.to(device)
    return pack


def discover_checkpoints(ckpt: str) -> List[str]:
    """
    Если ckpt — файл: вернуть [ckpt].
    Если ckpt — директория: собрать все fold*_best.pt.
    """
    if os.path.isfile(ckpt):
        return [ckpt]
    if os.path.isdir(ckpt):
        files = []
        for name in os.listdir(ckpt):
            if re.match(r"fold\d+_best\.pt$", name):
                files.append(os.path.join(ckpt, name))
        files.sort()
        if not files:
            # Возьмём любые .pt как запасной вариант
            files = [os.path.join(ckpt, f) for f in os.listdir(ckpt) if f.endswith(".pt")]
            files.sort()
        return files
    raise FileNotFoundError(f"Checkpoint path not found: {ckpt}")


def build_model_for_dataset(ds: RadiationDataset, batch_size: int, device: torch.device, d_model: int, gnn_layers: int, dropout: float) -> Tuple[DoseConstantPredictor, DataLoader]:
    """
    Создаёт DataLoader (без стандартизации), по первому батчу строит модель с корректными размерностями.
    Возвращает (model, loader_for_infer_without_std).
    """
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)
    first_pack = next(iter(loader))
    sample = {
        "atom_batch": first_pack.atom_batch,
        "frag_batch": first_pack.frag_batch,
        "num_feats_mol": first_pack.num_feats_mol,
        "num_feats_solv": first_pack.num_feats_solv,
    }
    model = build_from_sample(sample, d_model=d_model, gnn_layers=gnn_layers, dropout=dropout, use_mamba_in_fuser=True).to(device)
    return model, loader


def load_stats_from_ckpt(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract standardization and transform stats from checkpoint payload with safe fallbacks."""
    stats = {}
    # molecular num standardizer (from SSL)
    stats["std_mol_mean"] = payload.get("std_mol_mean", payload.get("standardizer_mean", None))
    stats["std_mol_std"]  = payload.get("std_mol_std",  payload.get("standardizer_std", None))
    # solvent num standardizer (from supervised fold)
    stats["std_solv_mean"] = payload.get("std_solv_mean", None)
    stats["std_solv_std"]  = payload.get("std_solv_std", None)
    # target (log10 z) and concentration transforms
    stats["y_mean_log"] = payload.get("y_mean_log", None)
    stats["y_std_log"]  = payload.get("y_std_log", None)
    stats["conc_mean"]  = payload.get("conc_mean", None)
    stats["conc_std"]   = payload.get("conc_std", None)
    return stats


@torch.no_grad()
def infer_once(model: DoseConstantPredictor, loader: DataLoader, device: torch.device, inverse_transform_pred) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    model.eval()
    preds: List[np.ndarray] = []
    for pack in loader:
        pack = to_device(pack, device)
        out = model(
            atom_batch=pack.atom_batch,
            frag_batch=pack.frag_batch,
            num_feats_mol=pack.num_feats_mol,
            num_feats_solv=pack.num_feats_solv,
            conc=getattr(pack, "conc", None),
            ssl_mask_cfg=None,
            return_ssl=False,
        )
        z = out["pred"].squeeze(-1)
        preds.append(inverse_transform_pred(z).cpu().numpy())
    preds_np = np.concatenate(preds, axis=0)
    return preds_np, {}


@torch.no_grad()
def infer_with_embeddings(model: DoseConstantPredictor, loader: DataLoader, device: torch.device, inverse_transform_pred) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    model.eval()
    preds: List[np.ndarray] = []
    atom_list: List[np.ndarray] = []
    frag_list: List[np.ndarray] = []
    fused_list: List[np.ndarray] = []
    for pack in loader:
        pack = to_device(pack, device)
        out = model(
            atom_batch=pack.atom_batch,
            frag_batch=pack.frag_batch,
            num_feats_mol=pack.num_feats_mol,
            num_feats_solv=pack.num_feats_solv,
            conc=getattr(pack, "conc", None),
            ssl_mask_cfg=None,
            return_ssl=False,
        )
        z = out["pred"].squeeze(-1)
        preds.append(inverse_transform_pred(z).cpu().numpy())
        emb = model.infer_embeddings(pack.atom_batch, pack.frag_batch, pack.num_feats_mol, pack.num_feats_solv)
        atom_list.append(emb["atom_vec"].detach().cpu().numpy())
        frag_list.append(emb["frag_vec"].detach().cpu().numpy())
        fused_list.append(emb["fused_vec"].detach().cpu().numpy())
    preds_np = np.concatenate(preds, axis=0)
    embeds = {
        "atom_vec": np.concatenate(atom_list, axis=0),
        "frag_vec": np.concatenate(frag_list, axis=0),
        "fused_vec": np.concatenate(fused_list, axis=0),
    }
    return preds_np, embeds


def main():
    parser = argparse.ArgumentParser(description="Predict dose constant with trained model (single ckpt or ensemble).")
    parser.add_argument("--csv", type=str, required=True, help="Path to input CSV with columns: smiles, (solv_smiles), diel_const, conc_stand, (dc_stand)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to a fold checkpoint .pt or a directory with fold*_best.pt")
    parser.add_argument("--out_csv", type=str, default="predictions.csv", help="Where to save predictions CSV")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--gnn_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_embeddings", action="store_true", help="Additionally save atom/frag/fused embeddings to .npz")
    parser.add_argument("--embeddings_path", type=str, default="embeddings.npz")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Загружаем датасет
    ds = RadiationDataset(args.csv)

    # Поднимаем модель под размерности датасета (по первому батчу)
    model, base_loader = build_model_for_dataset(
        ds, batch_size=args.batch_size, device=device, d_model=args.d_model, gnn_layers=args.gnn_layers, dropout=args.dropout
    )

    # Собираем чекпоинты (один или несколько)
    ckpts = discover_checkpoints(args.ckpt)
    if not ckpts:
        raise RuntimeError(f"No checkpoints found under {args.ckpt}")

    # Для каждого чекпоинта делаем отдельный прогон и усредняем предсказания (энсембль)
    all_preds: List[np.ndarray] = []
    agg_embeds: Dict[str, List[np.ndarray]] = {"atom_vec": [], "frag_vec": [], "fused_vec": []} if args.save_embeddings else {}

    for i, ckpt_path in enumerate(ckpts, start=1):
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = payload.get("model", payload.get("model_state_dict", payload))

        # Загрузка весов
        missing, unexpected = model.load_state_dict(state, strict=False)
        if i == 1:
            if missing:
                print(f"[i] Missing {len(missing)} keys (showing first 10): {missing[:10]}")
            if unexpected:
                print(f"[i] Unexpected {len(unexpected)} keys (showing first 10): {unexpected[:10]}")
        print(f"[✓] Loaded checkpoint ({i}/{len(ckpts)}): {ckpt_path}")

        stats = load_stats_from_ckpt(payload)
        # Prepare tensors for standardization; fallbacks if missing
        std_mol_mean = torch.tensor(stats["std_mol_mean"], dtype=torch.float) if stats["std_mol_mean"] is not None else None
        std_mol_std  = torch.tensor(stats["std_mol_std"],  dtype=torch.float) if stats["std_mol_std"]  is not None else None
        std_solv_mean = torch.tensor(stats["std_solv_mean"], dtype=torch.float) if stats["std_solv_mean"] is not None else None
        std_solv_std  = torch.tensor(stats["std_solv_std"],  dtype=torch.float) if stats["std_solv_std"]  is not None else None
        y_mean_log = stats["y_mean_log"] if stats["y_mean_log"] is not None else 0.0
        y_std_log  = stats["y_std_log"]  if stats["y_std_log"]  is not None else 1.0
        conc_mean  = stats["conc_mean"]  if stats["conc_mean"]  is not None else 0.0
        conc_std   = stats["conc_std"]   if stats["conc_std"]   is not None else 1.0

        # Fallbacks: compute from dataset if checkpoint lacks stats
        if std_mol_mean is None or std_mol_std is None:
            X_mol = torch.stack([ds[i].num_feats_mol for i in range(len(ds))], dim=0).float()
            std_mol_mean = X_mol.mean(dim=0)
            std_mol_std  = X_mol.std(dim=0).clamp_min(1e-8)
            print("[!] std_mol_* not found in ckpt — fitted on input CSV (molecule).")
        if std_solv_mean is None or std_solv_std is None:
            X_solv = torch.stack([ds[i].num_feats_solv for i in range(len(ds))], dim=0).float()
            std_solv_mean = X_solv.mean(dim=0)
            std_solv_std  = X_solv.std(dim=0).clamp_min(1e-8)
            print("[!] std_solv_* not found in ckpt — fitted on input CSV (solvent).")

        @torch.no_grad()
        def transform_num_mol(x: torch.Tensor) -> torch.Tensor:
            return (x - std_mol_mean) / std_mol_std.clamp_min(1e-8)
        @torch.no_grad()
        def transform_num_solv(x: torch.Tensor) -> torch.Tensor:
            return (x - std_solv_mean) / std_solv_std.clamp_min(1e-8)
        @torch.no_grad()
        def transform_conc(x: torch.Tensor) -> torch.Tensor:
            return (x - conc_mean) / max(conc_std, 1e-8)
        @torch.no_grad()
        def inverse_transform_pred(z: torch.Tensor) -> torch.Tensor:
            y_log = z * y_std_log + y_mean_log
            return torch.pow(10.0, y_log) - 1e-9

        def collate_and_standardize(items):
            pack = collate_batch(items)
            pack.num_feats_mol = transform_num_mol(pack.num_feats_mol)
            pack.num_feats_solv = transform_num_solv(pack.num_feats_solv)
            if hasattr(pack, "conc") and pack.conc is not None:
                pack.conc = transform_conc(pack.conc)
            return pack

        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_and_standardize)

        model.eval()
        preds_all = []
        if args.save_embeddings:
            emb_atom, emb_frag, emb_fused = [], [], []
        for pack in loader:
            pack = to_device(pack, device)
            out = model(
                atom_batch=pack.atom_batch,
                frag_batch=pack.frag_batch,
                num_feats_mol=pack.num_feats_mol,
                num_feats_solv=pack.num_feats_solv,
                conc=getattr(pack, "conc", None),
                ssl_mask_cfg=None,
                return_ssl=False,
            )
            z = out["pred"].squeeze(-1)
            preds_all.append(inverse_transform_pred(z).cpu().numpy())
            if args.save_embeddings:
                emb = model.infer_embeddings(pack.atom_batch, pack.frag_batch, pack.num_feats_mol, pack.num_feats_solv)
                emb_atom.append(emb["atom_vec"].detach().cpu().numpy())
                emb_frag.append(emb["frag_vec"].detach().cpu().numpy())
                emb_fused.append(emb["fused_vec"].detach().cpu().numpy())
        preds_np = np.concatenate(preds_all, axis=0)
        if args.save_embeddings:
            embeds = {
                "atom_vec": np.concatenate(emb_atom, axis=0),
                "frag_vec": np.concatenate(emb_frag, axis=0),
                "fused_vec": np.concatenate(emb_fused, axis=0),
            }
            for k in agg_embeds.keys():
                agg_embeds[k].append(embeds[k])

        all_preds.append(preds_np)

    # Усреднение по чекпоинтам
    preds_ens = np.mean(np.stack(all_preds, axis=0), axis=0)

    # Готовим вывод
    df = ds.df.copy()
    df["pred_dc"] = preds_ens

    if "dc_stand" in df.columns:
        try:
            df["abs_error"] = (df["pred_dc"] - df["dc_stand"]).abs()
        except Exception:
            pass

    out_dir = os.path.dirname(args.out_csv) or "."
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[✓] Saved predictions to {args.out_csv}")

    # Сохранить эмбеддинги (усреднение по ckpt'ам)
    if args.save_embeddings:
        # усредняем эмбеддинги по чекпоинтам
        emb_out = {}
        for k, stacks in agg_embeds.items():
            emb_out[k] = np.mean(np.stack(stacks, axis=0), axis=0)  # [N, D]
        np.savez(args.embeddings_path, **emb_out)
        print(f"[✓] Saved embeddings to {args.embeddings_path}")


if __name__ == "__main__":
    main()