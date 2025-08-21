# models/fuser.py
from __future__ import annotations
from typing import Optional, Dict, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

try:
    # PyTorch 2.x has TransformerEncoder; used as a fallback when Mamba isn't available
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
    _HAS_TFORMER = True
except Exception:
    _HAS_TFORMER = False

# Reuse Mamba wrapper from atom encoder module
from .mol_mamba import MambaWrapper


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MambaTransformerFuser(nn.Module):
    """
    Объединяет [ATOM, FRAG, NUM] в последовательность токенов и кодирует её Mamba/Transformer'ом.
    Поддерживает masked-fusion SSL (маскирование токенов и реконструкция NUM-фич).
    """
    def __init__(
        self,
        d_model: int = 256,
        num_feat_dim: int = 2 + 11,   # 2 среды + 11 базовых RDKit дескрипторов (см. data.py)
        out_dim: int = 256,
        dropout: float = 0.1,
        use_mamba: bool = True,
        transformer_nlayers: int = 2,
        transformer_nheads: int = 8,
        transformer_ffn: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_feat_dim = num_feat_dim

        # Проекция числовых фич в токен NUM
        self.num_proj = nn.Sequential(
            nn.LayerNorm(num_feat_dim),
            nn.Linear(num_feat_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Токен [CLS]
        self.cls_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Type embeddings: 0=CLS, 1=ATOM, 2=FRAG, 3=NUM
        self.type_emb = nn.Embedding(4, d_model)
        # Позиционные эмбеддинги для 4 позиций
        self.pos_emb = nn.Embedding(4, d_model)

        self.dropout = nn.Dropout(dropout)

        # Encoder: Mamba if available, else TransformerEncoder
        self.use_mamba = use_mamba
        if use_mamba:
            self.encoder = MambaWrapper(d_model=d_model, dropout=dropout, mamba_kwargs=None)
            self.is_mamba = True
        else:
            if not _HAS_TFORMER:
                raise RuntimeError("Transformer fallback is unavailable in this environment.")
            layer = TransformerEncoderLayer(d_model=d_model, nhead=transformer_nheads,
                                            dim_feedforward=d_model * transformer_ffn, dropout=dropout,
                                            activation="gelu", batch_first=True)
            self.encoder = TransformerEncoder(layer, num_layers=transformer_nlayers)
            self.is_mamba = False

        # Проекция из CLS в out_dim для финального предсказателя
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim),
        )

        # Heads for SSL masked-fusion (реконструкция NUM)
        self.recon_num_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_feat_dim),
        )

        # Маскируемые токены — обучаемый вектор [MASK]
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

    def _build_tokens(
        self,
        atom_vec: Tensor,  # [B, D]
        frag_vec: Tensor,  # [B, D]
        num_feats: Tensor, # [B, F]
        mask_cfg: Optional[Dict[str, float]] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Формирует последовательность токенов [CLS, ATOM, FRAG, NUM] и применяет маскирование (если задано).
        Возвращает:
          X: [B, 4, D]
          mask_dict: флаги маскирования по модальностям (bool) формы [B]
        """
        B, D = atom_vec.size(0), atom_vec.size(1)
        assert frag_vec.size(0) == B and frag_vec.size(1) == D
        assert num_feats.size(0) == B and num_feats.size(1) == self.num_feat_dim

        num_tok = self.num_proj(num_feats)  # [B, D]

        # Базовые токены
        cls_tok = self.cls_emb.expand(B, -1, -1)         # [B,1,D]
        atom_tok = atom_vec.unsqueeze(1)                  # [B,1,D]
        frag_tok = frag_vec.unsqueeze(1)                  # [B,1,D]
        num_tok = num_tok.unsqueeze(1)                    # [B,1,D]

        X = torch.cat([cls_tok, atom_tok, frag_tok, num_tok], dim=1)  # [B,4,D]

        # Типовые эмбеддинги и позиционные
        type_ids = torch.tensor([0, 1, 2, 3], device=X.device).unsqueeze(0).expand(B, 4)  # [B,4]
        pos_ids  = torch.tensor([0, 1, 2, 3], device=X.device).unsqueeze(0).expand(B, 4)

        X = X + self.type_emb(type_ids) + self.pos_emb(pos_ids)
        X = self.dropout(X)

        mask_dict = {
            "mask_atom": torch.zeros(B, dtype=torch.bool, device=X.device),
            "mask_frag": torch.zeros(B, dtype=torch.bool, device=X.device),
            "mask_num":  torch.zeros(B, dtype=torch.bool, device=X.device),
        }

        # Маскирование с вероятностями
        if mask_cfg is not None:
            pa = float(mask_cfg.get("atom", 0.0))
            pf = float(mask_cfg.get("frag", 0.0))
            pn = float(mask_cfg.get("num",  0.0))
            if pa > 0.0:
                m = torch.rand(B, device=X.device) < pa
                X[m, 1, :] = self.mask_token
                mask_dict["mask_atom"] = m
            if pf > 0.0:
                m = torch.rand(B, device=X.device) < pf
                X[m, 2, :] = self.mask_token
                mask_dict["mask_frag"] = m
            if pn > 0.0:
                m = torch.rand(B, device=X.device) < pn
                X[m, 3, :] = self.mask_token
                mask_dict["mask_num"] = m

        return X, mask_dict

    def forward(
        self,
        atom_vec: Tensor,   # [B, D_model]
        frag_vec: Tensor,   # [B, D_model]
        num_feats: Tensor,  # [B, F]
        mask_cfg: Optional[Dict[str, float]] = None,
        return_aux: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Возвращает словарь:
          - fused: [B, out_dim] – спроецированный CLS
          - cls:   [B, D]       – CLS после энкодера
          - tokens: [B, 4, D]   – все токены после энкодера
          - recon_num: [B, F]   – реконструкция NUM‑фич (для SSL); присутствует всегда, но считать лосс только по mask_num
          - mask_atom/frag/num: [B] bool – какие батчи были замаскированы
        """
        X, mask_dict = self._build_tokens(atom_vec, frag_vec, num_feats, mask_cfg)  # [B,4,D]

        if self.use_mamba:
            # MambaWrapper игнорирует маски, длина у нас фиксированная (4)
            Y = self.encoder(X, attn_mask=None)  # [B,4,D]
        else:
            # TransformerEncoder: создадим attn_mask, где 0=видимо; 1=маскировать (не нужно при фикс. полной длине)
            Y = self.encoder(X)  # [B,4,D]

        cls = Y[:, 0, :]  # [B,D]
        fused = self.out_proj(cls)  # [B,out_dim]

        # Реконструкция NUM‑фич из NUM‑токена
        num_hidden = Y[:, 3, :]  # [B,D]
        recon_num = self.recon_num_head(num_hidden)  # [B, F]

        out = {
            "fused": fused,
            "cls": cls,
            "tokens": Y,
            "recon_num": recon_num,
            "mask_atom": mask_dict["mask_atom"],
            "mask_frag": mask_dict["mask_frag"],
            "mask_num":  mask_dict["mask_num"],
        }

        if return_aux:
            # Для внешних SSL‑глав: отдаём скрытые ATOM/FRAG токены
            out.update({
                "atom_hidden": Y[:, 1, :],  # [B,D]
                "frag_hidden": Y[:, 2, :],  # [B,D]
            })
        return out


# -----------------------------
# CORAL loss (для Structural Distribution Alignment)
# -----------------------------
def coral_loss(h1: torch.Tensor, h2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    CORAL with per-dim standardization → scale-invariant & numerically stable.
    """
    b1, d = h1.size()
    b2, d2 = h2.size()
    if d != d2 or b1 <= 1 or b2 <= 1:
        return h1.new_tensor(0.0)
    def _std(x: torch.Tensor) -> torch.Tensor:
        xm = x - x.mean(dim=0, keepdim=True)
        xs = xm / (xm.std(dim=0, keepdim=True).clamp_min(eps))
        return xs
    h1n = _std(h1)
    h2n = _std(h2)
    c1 = (h1n.t() @ h1n) / (b1 - 1)
    c2 = (h2n.t() @ h2n) / (b2 - 1)
    return (c1 - c2).pow(2).mean()