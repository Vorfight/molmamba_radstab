# models/predictor.py
from __future__ import annotations
from typing import Optional, Dict, Any

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .mol_mamba import MolMambaEncoder
from .frag_mamba import FragEncoder
from .fuser import MambaTransformerFuser, coral_loss


class DoseConstantPredictor(nn.Module):
    """
    Полный стек:
      - atom_encoder: MolMambaEncoder (GNN -> node ordering -> Mamba/GRU -> pooling)
      - frag_encoder: FragEncoder (GraphConv stack -> global pooling)
      - fuser:       MambaTransformerFuser (tokens: [CLS, ATOM, FRAG, NUM_MOL, NUM_SOLV])
      - head:        регрессия в 1 значение (dose constant)

    Параметры:
      atom_in_dim, atom_edge_dim: размерности узлов и ребер атомного графа (из data.py)
      frag_in_dim: размерность узла фрагментного графа (RDKit дескрипторы фрагмента)
      num_feat_dim_mol: размерность числовых фич молекулы
      num_feat_dim_solv: размерность числовых фич растворителя
    """
    def __init__(
        self,
        atom_in_dim: int,
        atom_edge_dim: int,
        frag_in_dim: int,
        num_feat_dim_mol: int,
        num_feat_dim_solv: int,
        d_model: int = 256,
        gnn_layers: int = 3,
        dropout: float = 0.1,
        use_mamba_in_fuser: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_feat_dim_mol = num_feat_dim_mol
        self.num_feat_dim_solv = num_feat_dim_solv

        # 1) Энкодеры графов
        self.atom_encoder = MolMambaEncoder(
            in_node_dim=atom_in_dim,
            edge_dim=atom_edge_dim,
            hidden_dim=d_model,
            gnn_layers=gnn_layers,
            out_dim=d_model,
            mamba_kwargs=None,
            dropout=dropout,
        )
        self.frag_encoder = FragEncoder(
            in_node_dim=frag_in_dim,
            hidden_dim=d_model,
            gnn_layers=gnn_layers,
            out_dim=d_model,
            dropout=dropout,
        )

        # 2) Фьюзер
        self.fuser = MambaTransformerFuser(
            d_model=d_model,
            num_feat_dim_mol=num_feat_dim_mol,
            num_feat_dim_solv=num_feat_dim_solv,
            out_dim=d_model,
            dropout=dropout,
            use_mamba=use_mamba_in_fuser,
        )

        # 3) Encoder for concentration (scalar) and regression head conditioned on it
        self.conc_mlp = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    @torch.no_grad()
    def infer_embeddings(
        self,
        atom_batch,
        frag_batch,
        num_feats_mol: Tensor,
        num_feats_solv: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Возвращает промежуточные представления без градиентов:
          - atom_vec, frag_vec, fused_vec (CLS после фьюзера)
        """
        self.eval()
        atom_vec = self.atom_encoder(atom_batch)   # [B, D]
        frag_vec = self.frag_encoder(frag_batch)   # [B, D]
        fused_out = self.fuser(
            atom_vec=atom_vec,
            frag_vec=frag_vec,
            num_feats_mol=num_feats_mol,
            num_feats_solv=num_feats_solv,
            mask_cfg=None,
            return_aux=False,
        )
        fused_vec = fused_out["fused"]             # [B, D]
        return {"atom_vec": atom_vec, "frag_vec": frag_vec, "fused_vec": fused_vec}

    def forward(
        self,
        atom_batch,
        frag_batch,
        num_feats_mol: Tensor,                 # [B, F_mol]
        num_feats_solv: Tensor,                # [B, F_solv]
        conc: Optional[Tensor] = None,         # [B, 1] standardized concentration (separate input)
        ssl_mask_cfg: Optional[Dict] = None,   # e.g., {"atom":0.15, "frag":0.15, "num_mol":0.15}
        return_ssl: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Возвращает:
          - pred: [B, 1]
          - (если return_ssl) дополнительные поля из фьюзера для SSL
        """
        atom_vec = self.atom_encoder(atom_batch)  # [B,D]
        frag_vec = self.frag_encoder(frag_batch)  # [B,D]

        fuser_out = self.fuser(
            atom_vec=atom_vec,
            frag_vec=frag_vec,
            num_feats_mol=num_feats_mol,
            num_feats_solv=num_feats_solv,
            mask_cfg=ssl_mask_cfg,
            return_aux=return_ssl,
        )
        fused = fuser_out["fused"]                # [B,D]

        # Concentration as a separate conditioning input (if not provided, use zeros)
        B = num_feats_mol.size(0)
        if conc is None:
            conc = num_feats_mol.new_zeros(B, 1)
        conc_emb = self.conc_mlp(conc)           # [B,D]

        fused_plus = torch.cat([fused, conc_emb], dim=-1)  # [B, 2D]
        pred = self.head(fused_plus)              # [B,1]

        out = {"pred": pred}
        if return_ssl:
            out.update(fuser_out)
        return out

    @staticmethod
    def ssl_losses(
        fuser_out: Dict[str, Tensor],
        ssl_weights: Dict[str, float] = None,
    ) -> Dict[str, Tensor]:
        """
        Подсчёт SSL-лоссов:
          - masked NUM reconstruction (MSE, только по тем, где mask_num=True)
          - CORAL(ATOM, FRAG) — выравнивание распределений скрытых токенов
        Аргументы:
          fuser_out — результат forward(..., return_ssl=True)
          ssl_weights — веса лоссов, например {"num_recon": 1.0, "coral": 0.1}
        """
        if ssl_weights is None:
            ssl_weights = {"num_recon": 1.0, "coral": 0.1}

        losses = {}
        # use recon_num_mol if available; otherwise create a tensor on the same device via atom_hidden
        base_t = fuser_out.get("recon_num_mol", None)
        if base_t is None:
            base_t = fuser_out.get("atom_hidden", None)
        if base_t is None:
            base_t = torch.tensor(0.0)
        total = base_t.new_tensor(0.0)

        # 1) Masked NUM reconstruction (molecule-only). Actual MSE computed in trainer via masked_num_recon_loss.
        losses["num_recon"] = total.clone()

        # 2) CORAL между скрытыми представлениями ATOM и FRAG (если они доступны)
        if ("atom_hidden" in fuser_out) and ("frag_hidden" in fuser_out):
            coral = coral_loss(fuser_out["atom_hidden"], fuser_out["frag_hidden"])
            losses["coral"] = coral * float(ssl_weights.get("coral", 0.1))
            total = total + losses["coral"]
        else:
            losses["coral"] = total.clone()

        losses["total"] = total + losses["num_recon"]
        return losses

    @staticmethod
    def masked_num_recon_loss(
        recon_num_mol: Tensor,      # [B, F_mol] из fuser_out
        num_feats_mol_gt: Tensor,   # [B, F_mol] GT (standardized)
        mask_num_mol: Tensor,       # [B] bool из fuser_out
        weight: float = 1.0,
    ) -> Tensor:
        """
        MSE по молекулярной части NUM только для замаскированных примеров.
        """
        if mask_num_mol is None or mask_num_mol.numel() == 0:
            return recon_num_mol.new_tensor(0.0)
        if mask_num_mol.sum() == 0:
            return recon_num_mol.new_tensor(0.0)
        diff = recon_num_mol[mask_num_mol] - num_feats_mol_gt[mask_num_mol]
        return weight * F.mse_loss(diff, torch.zeros_like(diff), reduction="mean")


def build_from_sample(
    sample: Dict[str, Any],
    d_model: int = 256,
    gnn_layers: int = 3,
    dropout: float = 0.1,
    use_mamba_in_fuser: bool = True,
) -> DoseConstantPredictor:
    """
    Удобная фабрика: по одному батчу определяет нужные размерности.
    Ожидает словарь с ключами:
      - "atom_batch": PyG Batch (из collate_batch)
      - "frag_batch": PyG Batch
      - "num_feats_mol":  Tensor[B, F_mol]
      - "num_feats_solv": Tensor[B, F_solv]
    """
    atom_batch = sample["atom_batch"]
    frag_batch = sample["frag_batch"]
    num_feats_mol = sample["num_feats_mol"]
    num_feats_solv = sample["num_feats_solv"]

    atom_in_dim = int(atom_batch.x.size(1))
    atom_edge_dim = int(atom_batch.edge_attr.size(1))
    frag_in_dim = int(frag_batch.x.size(1))
    num_feat_dim_mol = int(num_feats_mol.size(1))
    num_feat_dim_solv = int(num_feats_solv.size(1))

    model = DoseConstantPredictor(
        atom_in_dim=atom_in_dim,
        atom_edge_dim=atom_edge_dim,
        frag_in_dim=frag_in_dim,
        num_feat_dim_mol=num_feat_dim_mol,
        num_feat_dim_solv=num_feat_dim_solv,
        d_model=d_model,
        gnn_layers=gnn_layers,
        dropout=dropout,
        use_mamba_in_fuser=use_mamba_in_fuser,
    )
    return model