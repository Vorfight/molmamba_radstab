# models/mol_mamba.py
from __future__ import annotations
from typing import List, Tuple, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv, global_mean_pool

# -----------------------------
# Optional Mamba import (fallback to GRU if not available)
# -----------------------------
_HAS_MAMBA = False
try:
    # pip install mamba-ssm
    from mamba_ssm import Mamba
    _HAS_MAMBA = True
except Exception:
    _HAS_MAMBA = False


# -----------------------------
# GNN for atom graph
# -----------------------------
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


class AtomGNN(nn.Module):
    """
    Stacked GINE with edge attributes.
    """
    def __init__(
        self,
        in_node_dim: int,
        edge_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj_in = nn.Linear(in_node_dim, hidden_dim)

        self.edge_mlp = MLP(edge_dim, hidden_dim, hidden_dim, dropout=dropout)

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            conv = GINEConv(
                nn=MLP(hidden_dim, hidden_dim, hidden_dim, dropout=dropout),
                train_eps=True,
            )
            self.layers.append(conv)
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        h = self.proj_in(x)
        e = self.edge_mlp(edge_attr)

        for conv, bn in zip(self.layers, self.norms):
            h_res = h
            h = conv(h, edge_index, e)
            h = bn(h)
            h = F.relu(h, inplace=True)
            h = self.dropout(h)
            h = h + h_res  # residual

        return h  # [num_nodes, hidden_dim]


# -----------------------------
# Deterministic node ordering per graph (BFS from max-degree node)
# -----------------------------
def _per_graph_slices(batch_vec: Tensor) -> List[Tuple[int, int]]:
    """
    Given batch vector of size [N] with values in [0..B-1], return (start, end) slices for each graph.
    """
    B = int(batch_vec.max().item()) + 1 if batch_vec.numel() > 0 else 0
    slices: List[Tuple[int, int]] = []
    for g in range(B):
        idx = (batch_vec == g).nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            slices.append((0, 0))
        else:
            start = int(idx.min().item())
            end = int(idx.max().item()) + 1
            slices.append((start, end))
    return slices


def _compute_degree(edge_index: Tensor, num_nodes: int) -> Tensor:
    deg = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), dtype=torch.long, device=edge_index.device))
    return deg


def _bfs_order(edge_index_local: Tensor, num_nodes: int, start_node: int) -> List[int]:
    """
    BFS on [0..num_nodes-1] using local edge_index (2, E_local).
    """
    adj = [[] for _ in range(num_nodes)]
    src, dst = edge_index_local
    for s, d in zip(src.tolist(), dst.tolist()):
        adj[s].append(d)

    visited = [False] * num_nodes
    order: List[int] = []
    q = [start_node]
    visited[start_node] = True
    while q:
        u = q.pop(0)
        order.append(u)
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                q.append(v)
    # If graph disconnected, append remaining nodes in ascending degree order
    if len(order) < num_nodes:
        remain = [i for i in range(num_nodes) if not visited[i]]
        order += remain
    return order


def node_ordering_for_batch(batch: Batch) -> Tuple[Tensor, Tensor, List[int]]:
    """
    Returns:
      perm:   [N] permutation indices to reorder nodes globally into per-graph sequences
      iperm:  [N] inverse permutation to restore original order
      lengths: list of lengths per graph
    """
    x = batch.x
    edge_index = batch.edge_index
    batch_vec = batch.batch  # [N]

    slices = _per_graph_slices(batch_vec)
    deg_all = _compute_degree(edge_index, x.size(0))

    perm_list: List[int] = []
    lengths: List[int] = []

    for g, (start, end) in enumerate(slices):
        if end <= start:
            lengths.append(0)
            continue
        # nodes of this graph in global index space
        nodes = torch.arange(start, end, device=x.device)
        # mask edges for this graph
        mask = (batch_vec[edge_index[0]] == g) & (batch_vec[edge_index[1]] == g)
        ei = edge_index[:, mask]
        # remap to local [0..n-1]
        local_map = {int(n.item()): i for i, n in enumerate(nodes)}
        src = ei[0].tolist()
        dst = ei[1].tolist()
        src_local = torch.tensor([local_map[s] for s in src], device=x.device)
        dst_local = torch.tensor([local_map[d] for d in dst], device=x.device)
        ei_local = torch.stack([src_local, dst_local], dim=0)

        # choose start node = max-degree; tie-breaker: min original index
        deg_local = _compute_degree(ei_local, num_nodes=end - start)
        start_local = int(torch.argmax(deg_local).item())

        order_local = _bfs_order(ei_local, num_nodes=end - start, start_node=start_local)
        order_global = (torch.tensor(order_local, device=x.device) + start).tolist()

        perm_list += order_global
        lengths.append(end - start)

    perm = torch.tensor(perm_list, device=x.device, dtype=torch.long)
    iperm = torch.empty_like(perm)
    iperm[perm] = torch.arange(perm.numel(), device=x.device, dtype=torch.long)
    return perm, iperm, lengths


# -----------------------------
# Mamba (or GRU) sequence encoder
# -----------------------------
class MambaWrapper(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, mamba_kwargs: Optional[dict] = None):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        if _HAS_MAMBA:
            kwargs = dict(d_model=d_model, d_state=16, d_conv=4, expand=2)
            if mamba_kwargs:
                kwargs.update(mamba_kwargs)
            self.seq = Mamba(**kwargs)
            self.is_mamba = True
        else:
            # Fallback: single-layer GRU with same hidden size
            self.seq = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)
            self.is_mamba = False

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        """
        x: [B, T, d_model]
        attn_mask: [B, T] (1 for valid, 0 for pad). For Mamba it's unused, for GRU we pack manually.
        """
        x = self.dropout(x)
        if self.is_mamba:
            # Mamba expects [B, T, d]
            out = self.seq(x)  # [B, T, d]
            return out
        else:
            if attn_mask is None:
                out, _ = self.seq(x)
                return out
            # pack padded sequence for GRU
            lengths = attn_mask.sum(dim=1).to(torch.int64).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths=lengths, batch_first=True, enforce_sorted=False)
            out_packed, _ = self.seq(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
            return out


# -----------------------------
# Mol-Mamba encoder: graph -> sequence -> pooled embedding
# -----------------------------
class MolMambaEncoder(nn.Module):
    """
    1) AtomGNN produces node embeddings.
    2) Nodes are deterministically ordered per graph (BFS from max-degree).
    3) Sequence encoder (Mamba/GRU) over ordered nodes.
    4) Pool: concat(last_valid, mean_valid, global_mean_pool) -> Linear -> out_dim.
    """
    def __init__(
        self,
        in_node_dim: int,
        edge_dim: int,
        hidden_dim: int = 256,
        gnn_layers: int = 3,
        out_dim: int = 256,
        mamba_kwargs: Optional[dict] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gnn = AtomGNN(
            in_node_dim=in_node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=gnn_layers,
            dropout=dropout,
        )
        self.order_pe = nn.Embedding(1024, hidden_dim)  # positional embedding by order idx (cap at 1024)
        self.mamba = MambaWrapper(d_model=hidden_dim, dropout=dropout, mamba_kwargs=mamba_kwargs)
        self.proj_out = nn.Linear(hidden_dim * 3, out_dim)
        self.dropout = nn.Dropout(dropout)

    def _pack_sequences(self, h: Tensor, batch: Batch, perm: Tensor, lengths: List[int]) -> Tuple[Tensor, Tensor]:
        """
        Reorder node embeddings by perm, split per graph, pad to max length.
        Returns:
          X_pad: [B, T_max, H]
          mask:  [B, T_max] (1=valid, 0=pad)
        """
        h_ord = h[perm]  # [N, H]
        # split
        B = len(lengths)
        chunks: List[Tensor] = []
        mask_list: List[Tensor] = []
        offset = 0
        max_len = max(lengths) if lengths else 0
        for L in lengths:
            if L == 0:
                chunks.append(torch.zeros(0, h.size(-1), device=h.device))
                mask_list.append(torch.zeros(0, dtype=torch.bool, device=h.device))
            else:
                chunks.append(h_ord[offset: offset + L])
                mask_list.append(torch.ones(L, dtype=torch.bool, device=h.device))
            offset += L
        # pad
        X_pad = pad_sequence(chunks, batch_first=True)  # [B, T_max, H]
        M_pad = pad_sequence(mask_list, batch_first=True)  # [B, T_max]
        # positional embedding by order index
        T = X_pad.size(1)
        pos_ids = torch.arange(T, device=h.device).clamp_max(1023)
        pos = pos_ids.unsqueeze(0).expand(B, T)  # [B, T]
        X_pad = X_pad + self.order_pe(pos)  # add order embedding
        return X_pad, M_pad

    def forward(self, batch: Batch) -> Tensor:
        """
        batch: PyG Batch with fields x, edge_index, edge_attr, batch
        returns: [B, out_dim]
        """
        x, edge_index, edge_attr, bvec = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        # 1) Atom GNN
        h = self.gnn(x, edge_index, edge_attr)  # [N, H]

        # 2) Node ordering
        perm, iperm, lengths = node_ordering_for_batch(batch)

        # 3) Pack sequences + order positional embedding
        X_pad, mask = self._pack_sequences(h, batch, perm, lengths)  # [B, T, H], [B, T]

        # 4) Sequence encoder (Mamba/GRU)
        seq_out = self.mamba(X_pad, attn_mask=mask)  # [B, T, H]

        # 5) Pool
        # last valid token per graph
        lengths_t = torch.tensor(lengths, device=seq_out.device, dtype=torch.long)
        last_idx = (lengths_t - 1).clamp_min(0)
        B, T, H = seq_out.size()
        gather_idx = last_idx.view(B, 1, 1).expand(B, 1, H)
        last_tok = seq_out.gather(1, gather_idx).squeeze(1)  # [B, H]

        # mean over valid tokens
        mask_f = mask.float()  # [B, T]
        sum_tok = (seq_out * mask_f.unsqueeze(-1)).sum(dim=1)  # [B, H]
        denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_tok = sum_tok / denom  # [B, H]

        # global mean pooling on GNN embeddings (orderless)
        gmp = global_mean_pool(h, bvec)  # [B, H]

        fused = torch.cat([last_tok, mean_tok, gmp], dim=-1)  # [B, 3H]
        fused = self.dropout(fused)
        out = self.proj_out(fused)  # [B, out_dim]
        return out