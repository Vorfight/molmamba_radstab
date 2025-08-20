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
    pack.num_feats = pack.num_feats.to(device)
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


def build_model_for_dataset(ds: RadiationDataset, batch_size: int, device: torch.device, d_model: int, gnn_layers: int, dropout: float) -> Tuple[DoseConstantPredictor, FeatureStandardizer, DataLoader]:
    """
    Создаёт DataLoader (без стандартизации), по первому батчу строит модель с корректными размерностями.
    Возвращает (model, standardizer(fitted? no), loader_for_infer_without_std).
    """
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)
    first_pack = next(iter(loader))
    sample = {
        "atom_batch": first_pack.atom_batch,
        "frag_batch": first_pack.frag_batch,
        "num_feats": first_pack.num_feats,
    }
    model = build_from_sample(sample, d_model=d_model, gnn_layers=gnn_layers, dropout=dropout, use_mamba_in_fuser=True).to(device)
    stdzr = FeatureStandardizer()
    return model, stdzr, loader


def load_std_from_ckpt(stdzr: FeatureStandardizer, payload: Dict[str, Any]) -> bool:
    mean_np = payload.get("standardizer_mean", None)
    std_np = payload.get("standardizer_std", None)
    if mean_np is None or std_np is None:
        return False
    stdzr.mean_ = torch.tensor(mean_np, dtype=torch.float)
    stdzr.std_ = torch.tensor(std_np, dtype=torch.float)
    return True


@torch.no_grad()
def infer_once(model: DoseConstantPredictor, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    model.eval()
    preds: List[np.ndarray] = []
    # Для эмбеддингов (если включено позже)
    atom_list: List[np.ndarray] = []
    frag_list: List[np.ndarray] = []
    fused_list: List[np.ndarray] = []

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
        preds.append(pred)

    preds_np = np.concatenate(preds, axis=0)
    embeds = {}
    return preds_np, embeds


@torch.no_grad()
def infer_with_embeddings(model: DoseConstantPredictor, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    model.eval()
    preds: List[np.ndarray] = []
    atom_list: List[np.ndarray] = []
    frag_list: List[np.ndarray] = []
    fused_list: List[np.ndarray] = []
    for pack in loader:
        pack = to_device(pack, device)
        # предсказание
        out = model(
            atom_batch=pack.atom_batch,
            frag_batch=pack.frag_batch,
            num_feats=pack.num_feats,
            ssl_mask_cfg=None,
            return_ssl=False,
        )
        pred = out["pred"].squeeze(-1).detach().cpu().numpy()
        preds.append(pred)
        # эмбеддинги
        emb = model.infer_embeddings(pack.atom_batch, pack.frag_batch, pack.num_feats)
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
    model, stdzr, base_loader = build_model_for_dataset(
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
        payload = torch.load(ckpt_path, map_location="cpu")
        state = payload.get("model_state_dict", payload)

        # Загрузка весов
        missing, unexpected = model.load_state_dict(state, strict=False)
        if i == 1:
            if missing:
                print(f"[i] Missing {len(missing)} keys (showing first 10): {missing[:10]}")
            if unexpected:
                print(f"[i] Unexpected {len(unexpected)} keys (showing first 10): {unexpected[:10]}")
        print(f"[✓] Loaded checkpoint ({i}/{len(ckpts)}): {ckpt_path}")

        # Стандартизатор: из чекпоинта или fit по всему входному CSV (fallback)
        ok_std = load_std_from_ckpt(stdzr, payload)
        if not ok_std:
            # fallback: fit на inference‑датасете (лучше, чем ничего)
            stdzr.fit(ds)
            print("[!] Standardizer stats not found in ckpt — fitted on input CSV.")

        # Обернём collate для применения стандартизации
        def collate_and_standardize(items):
            pack = collate_batch(items)
            pack.num_feats = stdzr.transform(pack.num_feats)
            return pack

        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_and_standardize)

        # Инференс
        if args.save_embeddings:
            preds_np, embeds = infer_with_embeddings(model, loader, device)
            for k in agg_embeds.keys():
                agg_embeds[k].append(embeds[k])
        else:
            preds_np, _ = infer_once(model, loader, device)

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