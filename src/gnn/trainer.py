"""
trainer.py - Trust learning trainer implementing Algorithm 1 from the paper.
Manages embedding learning, trust prediction, and saving results.

FIXES applied vs the original prototype:
  * The original training loop called `self.model(X, neighborhoods_idx)`
    INSIDE the mini-batch loop, i.e. it recomputed the *entire* forward pass
    (every node, every metapath, every layer) once per mini-batch instead of
    once per epoch. For a full-batch/transductive GNN like this one, that
    makes training cost scale with (num_pairs / batch_size) extra forward
    passes for no benefit. Fixed to do exactly one forward pass per epoch.
  * `predict_relations` looped over every edge in Python calling
    `model.predict_trust` one pair at a time. Replaced with a single
    vectorized `predict_trust_batch` call.
Author: H. Mezni (original) / fixes for Colab execution
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import os
from typing import Tuple, Dict, List

logger = logging.getLogger(__name__)


class TrustTrainer:
    """
    Orchestrates Algorithm 1:
    1. Initialize node embeddings from features
    2. Metapath-guided neighborhood sampling (done by MetapathSampler)
    3. Attention-based aggregation (GNN layers)
    4. Trust prediction and classification (Delta/Gamma sets)
    """

    def __init__(self, model, tkg, sampler, config: dict):
        self.model = model
        self.tkg = tkg
        self.sampler = sampler
        self.config = config
        self.embed_dim = config.get("embed_dim", 128)
        self.lr = config.get("learning_rate", 0.001)
        self.weight_decay = config.get("weight_decay", 1e-4)
        self.epochs = config.get("epochs", 200)
        self.batch_size = config.get("batch_size", 256)
        self.theta = config.get("trust_threshold", 0.6)
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.loss_fn = nn.BCELoss()
        self.device = torch.device("cpu")

    def build_features(self, feature_dim: int = 16) -> torch.Tensor:
        """Build feature matrix X from TKG node features."""
        nodes = [self.tkg.index_node[i] for i in range(len(self.tkg.node_index))]
        X = np.stack([self.tkg.node_features(n, feature_dim) for n in nodes])
        return torch.tensor(X, dtype=torch.float32)

    def build_edge_labels(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build positive (trusted) and negative (conflict) edge samples for training.
        Positive: TRUST, SUPPORT, ALLIED edges -> label 1
        Negative: OPPOSE, CONFLICT edges -> label 0
        """
        pos_pairs, neg_pairs = [], []
        node_index = self.tkg.node_index

        for u, v, data in self.tkg.G.edges(data=True):
            if u not in node_index or v not in node_index:
                continue
            ui, vi = node_index[u], node_index[v]
            rel = data.get("relation", "")
            if rel in ("TRUST", "SUPPORT", "ALLIED"):
                pos_pairs.append((ui, vi))
            elif rel in ("OPPOSE", "CONFLICT", "NEUTRAL"):
                neg_pairs.append((ui, vi))

        # Balance: sample same number of negatives as positives
        min_size = min(len(pos_pairs), len(neg_pairs))
        if min_size == 0:
            min_size = max(len(pos_pairs), len(neg_pairs))

        import random
        random.shuffle(pos_pairs)
        random.shuffle(neg_pairs)
        pos_pairs = pos_pairs[:min_size]
        neg_pairs = neg_pairs[:min_size]

        pairs = pos_pairs + neg_pairs
        labels = [1.0] * len(pos_pairs) + [0.0] * len(neg_pairs)

        if not pairs:
            # Degenerate graph with no usable edges - avoid a crash downstream.
            return torch.zeros((0, 2), dtype=torch.long), torch.zeros((0,), dtype=torch.float32)

        pairs_t = torch.tensor(pairs, dtype=torch.long)
        labels_t = torch.tensor(labels, dtype=torch.float32)
        return pairs_t, labels_t

    def train(self, nbr_idx_stack: torch.Tensor, nbr_mask_stack: torch.Tensor) -> torch.Tensor:
        """
        Run the full training loop (Algorithm 1, lines 1-13).
        One forward pass per epoch (full-batch / transductive training);
        the edge-label pairs are only used to compute the loss, never to
        trigger another forward pass.

        nbr_idx_stack / nbr_mask_stack: (num_metapaths, N, K) padded
            neighborhoods, as produced by MetapathSampler.to_padded_tensors.
        Returns: final trust embedding H (N, embed_dim)
        """
        X = self.build_features(self.config.get("feature_dim", 16))
        pairs, labels = self.build_edge_labels()
        N = X.shape[0]

        logger.info(f"Training TrustMPGNN: {N} nodes, {len(pairs)} edge samples, {self.epochs} epochs ...")

        if len(pairs) == 0:
            logger.warning("No labeled edges available for training; returning untrained embeddings.")
            self.model.eval()
            with torch.no_grad():
                return self.model(X, nbr_idx_stack, nbr_mask_stack)

        best_loss = float("inf")
        best_H = None

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            self.optimizer.zero_grad()

            # Single forward pass for the whole graph this epoch.
            H = self.model(X, nbr_idx_stack, nbr_mask_stack)

            # Mini-batch the LOSS (not the forward pass) for memory friendliness
            # on very large pair sets, but reuse the same H/graph throughout.
            perm = torch.randperm(len(pairs))
            pairs_shuf = pairs[perm]
            labels_shuf = labels[perm]

            total_loss = 0.0
            n_chunks = 0
            for start in range(0, len(pairs_shuf), self.batch_size):
                batch_pairs = pairs_shuf[start:start + self.batch_size]
                batch_labels = labels_shuf[start:start + self.batch_size]
                scores = self.model.predict_trust_batch(H, batch_pairs)
                loss = self.loss_fn(scores, batch_labels)
                total_loss = total_loss + loss
                n_chunks += 1

            avg_loss = total_loss / max(n_chunks, 1)
            avg_loss.backward()
            self.optimizer.step()

            loss_val = avg_loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
                best_H = H.detach().clone()

            if epoch % max(1, self.epochs // 10) == 0 or epoch == 1:
                logger.info(f"  Epoch {epoch}/{self.epochs} | Loss: {loss_val:.4f}")

        logger.info(f"Training complete. Best loss: {best_loss:.4f}")
        return best_H

    def predict_relations(self, H: torch.Tensor) -> Tuple[list, list, list]:
        """
        Algorithm 1, lines 15-25: predict trust/conflict for all node pairs.
        Uses candidate pairs from existing edges only (for efficiency).
        Returns: E_hat (all predicted), Delta (trusted), Gamma (conflict).
        Vectorized: one batched forward through the predictor instead of a
        Python loop calling `predict_trust` edge-by-edge.
        """
        self.model.eval()
        node_index = self.tkg.node_index

        edges = [(u, v) for u, v, data in self.tkg.G.edges(data=True)
                 if u in node_index and v in node_index]
        if not edges:
            return [], [], []

        pairs = torch.tensor([[node_index[u], node_index[v]] for u, v in edges], dtype=torch.long)

        with torch.no_grad():
            scores = self.model.predict_trust_batch(H, pairs).cpu().numpy()

        E_hat, Delta, Gamma = [], [], []
        for (u, v), score in zip(edges, scores):
            score = float(score)
            triple = (u, score, v)
            E_hat.append(triple)
            if score >= self.theta:
                Delta.append(triple)
            else:
                Gamma.append(triple)

        logger.info(f"Trust prediction: {len(Delta)} trusted, {len(Gamma)} conflict pairs (theta={self.theta})")
        return E_hat, Delta, Gamma

    def save_embeddings(self, H: torch.Tensor, path: str):
        """Save trust embedding space to file."""
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        torch.save({"embeddings": H, "node_index": self.tkg.node_index}, path)
        logger.info(f"Embeddings saved to {path}")
