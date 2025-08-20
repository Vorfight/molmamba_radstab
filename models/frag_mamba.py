# models/frag_mamba.py
from __future__ import annotations
from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from torch_geometric.data import Batch
from torch_geometric.nn import GraphConv, global_mean_pool, global_max_pool


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


class FragGNN(nn.Module):
    """
    Стек GraphConv для фрагментного графа. Используем edge_attr как edge_weight (если есть).
    Узловые признаки — числовые RDKit-дескрипторы фрагментов.
    """
    def __init__(
        self,
        in_node_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_node_dim, hidden_dim)

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GraphConv(hidden_dim, hidden_dim, aggr="add", normalize=True))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor]) -> Tensor:
        h = self.in_proj(x)
        # Сожмём edge_attr до веса, если это [E,1]
        edge_weight = None
        if edge_attr is not None:
            if edge_attr.dim() == 2 and edge_attr.size(1) == 1:
                edge_weight = edge_attr.view(-1)
            elif edge_attr.dim() == 1:
                edge_weight = edge_attr
            # иначе игнорируем (GraphConv не поддерживает многомерные edge_attr)

        for conv, bn in zip(self.layers, self.norms):
            h_res = h
            h = conv(h, edge_index, edge_weight=edge_weight)
            h = bn(h)
            h = F.relu(h, inplace=True)
            h = self.dropout(h)
            h = h + h_res  # residual

        return h  # [num_nodes, hidden_dim]


class FragEncoder(nn.Module):
    """
    Фрагментный энкодер:
      1) GraphConv-энкодер узлов.
      2) Глобальный пуллинг: mean и max по узлам графа.
      3) Конкатенация [mean | max] -> Linear -> out_dim.
    """
    def __init__(
        self,
        in_node_dim: int,
        hidden_dim: int = 256,
        gnn_layers: int = 3,
        out_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gnn = FragGNN(
            in_node_dim=in_node_dim,
            hidden_dim=hidden_dim,
            num_layers=gnn_layers,
            dropout=dropout,
        )
        self.proj_out = nn.Linear(hidden_dim * 2, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch: Batch) -> Tensor:
        """
        batch: PyG Batch с полями x, edge_index, edge_attr, batch
        return: [B, out_dim]
        """
        x, edge_index, edge_attr, bvec = batch.x, batch.edge_index, batch.edge_attr, batch.batch

        # 1) Узловые эмбеддинги
        h = self.gnn(x, edge_index, edge_attr)  # [N, H]

        # 2) Глобальный пуллинг по графам
        h_mean = global_mean_pool(h, bvec)  # [B, H]
        h_max = global_max_pool(h, bvec)    # [B, H]

        # 3) Слияние и проекция
        fused = torch.cat([h_mean, h_max], dim=-1)  # [B, 2H]
        fused = self.dropout(fused)
        out = self.proj_out(fused)  # [B, out_dim]
        return out