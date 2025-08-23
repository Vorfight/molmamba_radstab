# ssl_pretrain.py
from __future__ import annotations
import os
import random
import argparse
from typing import Dict, Any

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from data import RadiationDataset, collate_batch, FeatureStandardizer
from models.predictor import DoseConstantPredictor, build_from_sample
from models.fuser import coral_loss


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch_pack, device: torch.device):
    batch_pack.atom_batch = batch_pack.atom_batch.to(device)
    batch_pack.frag_batch = batch_pack.frag_batch.to(device)
    if hasattr(batch_pack, "num_feats_mol") and batch_pack.num_feats_mol is not None:
        batch_pack.num_feats_mol = batch_pack.num_feats_mol.to(device)
    if hasattr(batch_pack, "num_feats_solv") and batch_pack.num_feats_solv is not None:
        batch_pack.num_feats_solv = batch_pack.num_feats_solv.to(device)
    if hasattr(batch_pack, "conc") and batch_pack.conc is not None:
        batch_pack.conc = batch_pack.conc.to(device)
    if batch_pack.y is not None:
        batch_pack.y = batch_pack.y.to(device)
    return batch_pack


def freeze_module(m: nn.Module, freeze: bool = True):
    for p in m.parameters():
        p.requires_grad = not freeze


def save_checkpoint(path: str, model: DoseConstantPredictor, stdzr: FeatureStandardizer, args: Dict[str, Any]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mol_mean = stdzr.mean_.cpu().numpy() if stdzr.mean_ is not None else None
    mol_std = stdzr.std_.cpu().numpy() if stdzr.std_ is not None else None
    payload = {
        "model_state_dict": model.state_dict(),
        # legacy keys (train.py can read these)
        "standardizer_mean": mol_mean,
        "standardizer_std": mol_std,
        # explicit keys for molecular standardizer (new API)
        "std_mol_mean": mol_mean,
        "std_mol_std": mol_std,
        "args": args,
    }
    torch.save(payload, path)
    print(f"[✓] Saved checkpoint to {path}")


# -----------------------------
# Stage 1: Structural Distribution Alignment (CORAL on encoders)
# -----------------------------
def run_stage1_sda(
    model: DoseConstantPredictor,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 2e-4,
    coral_weight: float = 1.0,
    grad_clip: float = 1.0,
):
    # Train only encoders (freeze fuser + head + conc_mlp)
    freeze_module(model.fuser, True)
    freeze_module(model.head, True)
    if hasattr(model, "conc_mlp"):
        freeze_module(model.conc_mlp, True)
    freeze_module(model.atom_encoder, False)
    freeze_module(model.frag_encoder, False)

    optim = torch.optim.AdamW(
        list(model.atom_encoder.parameters()) + list(model.frag_encoder.parameters()),
        lr=lr, weight_decay=1e-4
    )

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0

        for pack in loader:
            pack = to_device(pack, device)

            # Получаем эмбеддинги энкодеров без фьюзера
            atom_vec = model.atom_encoder(pack.atom_batch)   # [B, D]
            frag_vec = model.frag_encoder(pack.frag_batch)   # [B, D]

            # CORAL alignment
            if atom_vec.size(0) > 1 and frag_vec.size(0) > 1:
                loss = coral_loss(atom_vec, frag_vec) * coral_weight
            else:
                loss = atom_vec.new_tensor(0.0)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(list(model.atom_encoder.parameters()) + list(model.frag_encoder.parameters()), grad_clip)
            optim.step()

            total_loss += float(loss.item())
            n_batches += 1

        avg = total_loss / max(1, n_batches)
        print(f"[Stage1][Epoch {epoch:03d}] CORAL loss: {avg:.6f}")


# -----------------------------
# Stage 2: E‑Semantic Masked Fusion (reconstruct NUM)
# -----------------------------
def run_stage2_masked_fusion(
    model: DoseConstantPredictor,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 80,
    lr: float = 2e-4,
    num_recon_weight: float = 1.0,
    coral_hidden_weight: float = 0.1,
    mask_prob_atom: float = 0.0,
    mask_prob_frag: float = 0.0,
    mask_prob_num_mol: float = 0.15,
    mask_prob_num_solv: float = 0.0,
    grad_clip: float = 1.0,
):
    # Размораживаем всё, кроме финальной головы (не нужна в SSL)
    freeze_module(model.head, True)
    if hasattr(model, "conc_mlp"):
        freeze_module(model.conc_mlp, True)
    freeze_module(model.fuser, False)
    freeze_module(model.atom_encoder, False)
    freeze_module(model.frag_encoder, False)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Determine molecular numeric dimensionality (for SSL reconstruction of molecule-only NUM)
    num_mol_dim = None

    model.train()
    for epoch in range(1, epochs + 1):
        total_num = 0.0
        total_coral = 0.0
        n_batches = 0

        for pack in loader:
            pack = to_device(pack, device)

            # Маскирование модальностей для SSL
            mask_cfg = {
                "atom": mask_prob_atom,
                "frag": mask_prob_frag,
                "num_mol": mask_prob_num_mol,
                "num_solv": mask_prob_num_solv,
            }

            out = model(
                atom_batch=pack.atom_batch,
                frag_batch=pack.frag_batch,
                num_feats_mol=pack.num_feats_mol,
                num_feats_solv=pack.num_feats_solv,
                ssl_mask_cfg=mask_cfg,
                return_ssl=True,
            )

            # 1) MSE реконструкции ТОЛЬКО молекулярной части NUM (исключаем растворитель)
            recon_mol = out["recon_num_mol"]          # [B, F_mol]
            mask_num_mol = out.get("mask_num_mol", out.get("mask_num", None))  # [B] bool
            tgt_mol = pack.num_feats_mol               # standardized in collate

            if mask_num_mol is not None and mask_num_mol.sum() > 0:
                diff = recon_mol[mask_num_mol] - tgt_mol[mask_num_mol]
                loss_num = torch.mean(diff.pow(2)) * num_recon_weight
            else:
                loss_num = recon_mol.new_tensor(0.0)

            # 2) Доп. CORAL между скрытыми ATOM и FRAG токенами во фьюзере (слабый вес)
            if ("atom_hidden" in out) and ("frag_hidden" in out):
                ah = out["atom_hidden"]
                fh = out["frag_hidden"]
                if ah.size(0) > 1 and fh.size(0) > 1:
                    loss_coral = coral_loss(ah, fh) * coral_hidden_weight
                else:
                    loss_coral = ah.new_tensor(0.0)
            else:
                loss_coral = out["fused"].new_tensor(0.0)

            loss = loss_num + loss_coral

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()

            total_num += float(loss_num.item())
            total_coral += float(loss_coral.item())
            n_batches += 1

        avg_num = total_num / max(1, n_batches)
        avg_cor = total_coral / max(1, n_batches)
        print(f"[Stage2][Epoch {epoch:03d}] reconNUM_mol: {avg_num:.6f} | coralHidden: {avg_cor:.6f}")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Two-stage SSL pretraining: SDA + E-semantic masked fusion")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV with columns: smiles, solvent_smiles, diel_const, concentration, dose_constant")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs_sda", type=int, default=50)
    parser.add_argument("--epochs_mask", type=int, default=80)
    parser.add_argument("--lr_sda", type=float, default=2e-4)
    parser.add_argument("--lr_mask", type=float, default=2e-4)
    parser.add_argument("--num_recon_weight", type=float, default=1.0)
    parser.add_argument("--coral_hidden_weight", type=float, default=0.1)
    parser.add_argument("--mask_prob_atom", type=float, default=0.0)
    parser.add_argument("--mask_prob_frag", type=float, default=0.0)
    parser.add_argument("--mask_prob_num_mol", type=float, default=0.15)
    parser.add_argument("--mask_prob_num_solv", type=float, default=0.0)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--gnn_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", type=str, default="ssl_pretrained.pt")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = RadiationDataset(args.csv)

    stdzr = FeatureStandardizer()
    with torch.no_grad():
        X = torch.stack([ds[i].num_feats_mol for i in range(len(ds))], dim=0).float()
        stdzr.mean_ = X.mean(dim=0)
        stdzr.std_ = X.std(dim=0).clamp_min(1e-8)

    def collate_and_standardize(items):
        pack = collate_batch(items)
        pack.num_feats_mol = stdzr.transform(pack.num_feats_mol)
        # pack.num_feats_solv не трогаем в SSL
        return pack

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_and_standardize, drop_last=False
    )

    # Инициализация модели по одному батчу
    first_pack = next(iter(loader))
    sample = {
        "atom_batch": first_pack.atom_batch,
        "frag_batch": first_pack.frag_batch,
        "num_feats_mol": first_pack.num_feats_mol,
        "num_feats_solv": first_pack.num_feats_solv,
    }
    # Note: concentration is deliberately not passed here; SSL does not use it and the regression head is frozen.
    model = build_from_sample(
        sample=sample,
        d_model=args.d_model,
        gnn_layers=args.gnn_layers,
        dropout=args.dropout,
        use_mamba_in_fuser=True,
    ).to(device)

    # Stage 1: SDA
    run_stage1_sda(
        model=model,
        loader=loader,
        device=device,
        epochs=args.epochs_sda,
        lr=args.lr_sda,
        coral_weight=1.0,
        grad_clip=1.0,
    )

    # Stage 2: E‑Semantic Masked Fusion
    run_stage2_masked_fusion(
        model=model,
        loader=loader,
        device=device,
        epochs=args.epochs_mask,
        lr=args.lr_mask,
        num_recon_weight=args.num_recon_weight,
        coral_hidden_weight=args.coral_hidden_weight,
        mask_prob_atom=args.mask_prob_atom,
        mask_prob_frag=args.mask_prob_frag,
        mask_prob_num_mol=args.mask_prob_num_mol,
        mask_prob_num_solv=args.mask_prob_num_solv,
        grad_clip=1.0,
    )

    # Save checkpoint
    save_checkpoint(args.ckpt, model, stdzr, vars(args))


if __name__ == "__main__":
    main()