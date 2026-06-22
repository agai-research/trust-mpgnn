"""
model.py - Trust-MPGNN: Metapath-guided GNN for trust learning.
Implements Algorithm 1 (metapath-based trust learning) from the paper.

FIXES applied vs the original prototype:
  * MetapathAttentionLayer / TrustMPGNNLayer used a pure-Python `for v in range(N)`
    loop (one tensor op per node, per metapath, per layer, per epoch). This made
    training effectively unusable on anything bigger than a toy graph (a 2000+
    node TKG with the paper's defaults -> ~tens of millions of tiny Python-level
    tensor ops). Rewritten below using padded-neighbor tensors + masked, batched
    attention so the whole layer is a handful of vectorized tensor ops.
Author: H. Mezni (original) / fixes for Colab execution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MetapathAttentionLayer(nn.Module):
    """
    Single metapath-specific attention aggregation layer (vectorized).
    Computes attention weights e_{vu}^{(m)} and aggregates neighbor embeddings
    for ALL nodes at once using a padded neighbor-index tensor + mask.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        # Attention vector a_m (applied to concatenated source+neighbor projections)
        self.a = nn.Parameter(torch.empty(2 * out_dim))
        nn.init.xavier_uniform_(self.a.view(1, -1))
        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_dim

    def forward(self, h: torch.Tensor, nbr_idx: torch.Tensor, nbr_mask: torch.Tensor) -> torch.Tensor:
        """
        h: (N, in_dim) all node embeddings
        nbr_idx: (N, K) long tensor of neighbor indices (padded with 0 where invalid)
        nbr_mask: (N, K) bool tensor, True where the neighbor slot is valid
        Returns: (N, out_dim) aggregated embedding per node (self-projection used
                 for nodes that have no valid neighbors under this metapath).
        """
        N, K = nbr_idx.shape
        Wh = self.W(h)                     # (N, out_dim) - project every node once
        Wu = Wh[nbr_idx]                   # (N, K, out_dim) gather neighbor projections
        Wv_exp = Wh.unsqueeze(1).expand(-1, K, -1)          # (N, K, out_dim)
        concat = torch.cat([Wv_exp, Wu], dim=-1)            # (N, K, 2*out_dim)

        e = self.leaky(concat @ self.a)                     # (N, K)
        e = e.masked_fill(~nbr_mask, float("-inf"))

        has_any = nbr_mask.any(dim=1)                        # (N,)
        # Avoid NaNs from softmax over an all -inf row; replaced afterwards anyway.
        e_safe = torch.where(nbr_mask, e, torch.zeros_like(e))
        alpha = F.softmax(e_safe.masked_fill(~nbr_mask, float("-inf")), dim=1)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        alpha = self.dropout(alpha)

        out = (alpha.unsqueeze(-1) * Wu).sum(dim=1)          # (N, out_dim)
        # Nodes with no valid neighbors fall back to their own projection.
        out = torch.where(has_any.unsqueeze(-1), out, Wh)
        return out


class TrustMPGNNLayer(nn.Module):
    """
    One GNN layer across all metapaths.
    For each metapath m, computes attention-based aggregation, then combines.
    """

    def __init__(self, in_dim: int, out_dim: int, num_metapaths: int, dropout: float = 0.3):
        super().__init__()
        self.mp_layers = nn.ModuleList([
            MetapathAttentionLayer(in_dim, out_dim, dropout) for _ in range(num_metapaths)
        ])
        self.W_self = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.ELU()

    def forward(self, h: torch.Tensor, nbr_idx_stack: torch.Tensor, nbr_mask_stack: torch.Tensor) -> torch.Tensor:
        """
        h: (N, in_dim) all node embeddings
        nbr_idx_stack: (num_metapaths, N, K) padded neighbor indices
        nbr_mask_stack: (num_metapaths, N, K) validity mask
        Returns updated embeddings (N, out_dim).
        """
        mp_outs = []
        for mp_idx, mp_layer in enumerate(self.mp_layers):
            agg = mp_layer(h, nbr_idx_stack[mp_idx], nbr_mask_stack[mp_idx])
            mp_outs.append(agg)

        aggregated = torch.stack(mp_outs, dim=0).sum(dim=0)  # (N, out_dim)
        h_self = self.W_self(h)
        out = self.act(self.norm(aggregated + h_self))
        return out


class TrustPredictor(nn.Module):
    """
    MLP-based trust predictor: r_hat_uv = sigma(MLP(h_u || h_v))
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, h_u: torch.Tensor, h_v: torch.Tensor) -> torch.Tensor:
        """Predict trust score for pair (u, v). Shape: scalar or batch."""
        concat = torch.cat([h_u, h_v], dim=-1)
        return self.mlp(concat).squeeze(-1)


class TrustMPGNN(nn.Module):
    """
    Full Trust-MPGNN model: metapath-guided embedding + trust prediction.
    """

    def __init__(self, input_dim: int, embed_dim: int, num_layers: int,
                 num_metapaths: int, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.layers = nn.ModuleList([
            TrustMPGNNLayer(embed_dim, embed_dim, num_metapaths, dropout)
            for _ in range(num_layers)
        ])
        self.predictor = TrustPredictor(embed_dim)

    def forward(self, x: torch.Tensor, nbr_idx_stack: torch.Tensor, nbr_mask_stack: torch.Tensor) -> torch.Tensor:
        """
        x: (N, input_dim) node features
        nbr_idx_stack / nbr_mask_stack: (num_metapaths, N, K) padded neighborhoods
        Returns node embeddings H of shape (N, embed_dim).
        """
        h = F.relu(self.input_proj(x))
        for layer in self.layers:
            h = layer(h, nbr_idx_stack, nbr_mask_stack)
        return h

    def predict_trust(self, h: torch.Tensor, u: int, v: int) -> float:
        """Predict trust score for pair (u, v) given embeddings H."""
        return self.predictor(h[u], h[v]).item()

    def predict_trust_batch(self, h: torch.Tensor,
                             pairs: torch.Tensor) -> torch.Tensor:
        """
        pairs: (B, 2) tensor of (u, v) indices
        Returns (B,) trust scores.
        """
        h_u = h[pairs[:, 0]]
        h_v = h[pairs[:, 1]]
        return self.predictor(h_u, h_v)
