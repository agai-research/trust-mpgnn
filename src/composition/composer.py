"""
composer.py - Trust-aware IoT service selection and composition.
Implements Algorithm 2 from the Trust-MPGNN paper.
Author: H. Mezni
"""

import torch
import numpy as np
import logging
import json
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


def cosine_similarity_matrix(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity matrix between rows of A and B.
    S[i,j] = (A[i] . B[j]) / (||A[i]|| ||B[j]||)
    """
    A_norm = A / (A.norm(dim=1, keepdim=True) + 1e-8)
    B_norm = B / (B.norm(dim=1, keepdim=True) + 1e-8)
    return A_norm @ B_norm.T  # (|A|, |B|)


def find_best_service(H: torch.Tensor, F_S: torch.Tensor, task_vec: torch.Tensor,
                       trusted_service_idxs: List[int],
                       used_idxs: Optional[set] = None) -> int:
    """
    Find best IoT service for a task using cosine similarity between task_vec and service embeddings.
    Excludes already-used indices to promote diversity across workflow tasks.
    Returns global node index.
    """
    if len(trusted_service_idxs) == 0:
        return -1
    # Prefer unused candidates
    candidates = [i for i in trusted_service_idxs if used_idxs is None or i not in used_idxs]
    if not candidates:
        candidates = trusted_service_idxs

    H_candidates = H[candidates]                  # (n_c, embed_dim)
    embed_dim = H_candidates.shape[1]
    # Project task_vec to embed_dim if needed
    if task_vec.shape[0] != embed_dim:
        # Repeat/truncate to match embed_dim
        repeats = (embed_dim + task_vec.shape[0] - 1) // task_vec.shape[0]
        task_proj = task_vec.repeat(repeats)[:embed_dim]
    else:
        task_proj = task_vec
    task_norm = task_proj / (task_proj.norm() + 1e-8)
    H_norm = H_candidates / (H_candidates.norm(dim=1, keepdim=True) + 1e-8)
    scores = H_norm @ task_norm                   # (n_c,)
    best_local = scores.argmax().item()
    return candidates[best_local]


def find_best_resource(H: torch.Tensor, F_R: torch.Tensor, res_vec: torch.Tensor,
                        trusted_resource_idxs: List[int],
                        used_idxs: Optional[set] = None) -> int:
    """
    Find best IoT resource using cosine similarity. Excludes already-used resources.
    """
    if len(trusted_resource_idxs) == 0:
        return -1
    candidates = [i for i in trusted_resource_idxs if used_idxs is None or i not in used_idxs]
    if not candidates:
        candidates = trusted_resource_idxs

    H_candidates = H[candidates]
    embed_dim = H_candidates.shape[1]
    if res_vec.shape[0] != embed_dim:
        repeats = (embed_dim + res_vec.shape[0] - 1) // res_vec.shape[0]
        res_proj = res_vec.repeat(repeats)[:embed_dim]
    else:
        res_proj = res_vec
    res_norm = res_proj / (res_proj.norm() + 1e-8)
    H_norm = H_candidates / (H_candidates.norm(dim=1, keepdim=True) + 1e-8)
    scores = H_norm @ res_norm
    best_local = scores.argmax().item()
    return candidates[best_local]


class IoTComposer:
    """
    Trust-aware IoT service composition module.
    Uses precomputed trust embeddings H to select and compose IoT services/resources.
    """

    def __init__(self, tkg, H: torch.Tensor, Delta: list, Gamma: list, config: dict):
        """
        tkg: TrustKnowledgeGraph
        H: (N, d) trust embedding matrix
        Delta: trusted relation triples (u, score, v)
        Gamma: conflict relation triples
        config: composition configuration
        """
        self.tkg = tkg
        self.H = H
        self.Delta = Delta
        self.Gamma = Gamma
        self.config = config
        self.conflict_set = {(u, v) for u, score, v in Gamma}
        self.trust_set = {(u, v) for u, score, v in Delta}

        # Pre-sort node indices by type
        self.service_idxs = [tkg.node_index[sid] for sid in tkg.services
                              if sid in tkg.node_index]
        self.resource_idxs = [tkg.node_index[rid] for rid in tkg.resources
                               if rid in tkg.node_index]
        self.provider_idxs = [tkg.node_index[pid] for pid in tkg.providers
                               if pid in tkg.node_index]

        # Build trusted subsets (filter out high-conflict entities)
        self.trusted_service_idxs = self._filter_trusted(self.service_idxs)
        self.trusted_resource_idxs = self._filter_trusted(self.resource_idxs)

    def _filter_trusted(self, idxs: List[int]) -> List[int]:
        """
        Keep only entities that appear in Delta and NOT in Gamma conflict pairs.
        """
        index_node = self.tkg.index_node
        trusted = []
        for idx in idxs:
            node_id = index_node.get(idx)
            if node_id is None:
                continue
            in_conflict = any(
                (node_id == u or node_id == v) for u, v in self.conflict_set
            )
            if not in_conflict:
                trusted.append(idx)
        return trusted if trusted else idxs  # fallback to all if none remain

    def _proximity_degree(self, service_idx: int, resource_idx: int) -> float:
        """
        Compute proximity degree d: combines trust chain ratio with embedding similarity.
        d is used in the composition scoring function (Eq. score).
        """
        s_id = self.tkg.index_node.get(service_idx, "")
        r_id = self.tkg.index_node.get(resource_idx, "")
        trust_count = sum(1 for u, v in self.trust_set
                          if u in (s_id, r_id) or v in (s_id, r_id))
        conflict_count = sum(1 for u, v in self.conflict_set
                             if u in (s_id, r_id) or v in (s_id, r_id))
        total = trust_count + conflict_count
        if total == 0:
            # Fallback: use cosine similarity between embeddings
            if service_idx < self.H.shape[0] and resource_idx < self.H.shape[0]:
                hs = self.H[service_idx]
                hr = self.H[resource_idx]
                cos = float((hs @ hr) / (hs.norm() * hr.norm() + 1e-8))
                return max(0.0, min(1.0, (cos + 1.0) / 2.0))
            return 0.5
        return trust_count / total

    def compose(self, workflow: List[Dict]) -> Dict:
        """
        Algorithm 2: Trust-aware IoT Service Selection.
        workflow: list of {task_id, task_name, task_features, resource_features}
        Returns: composition result with assigned services, resources, and score.
        """
        logger.info(f"Composing IoT application for workflow of {len(workflow)} tasks ...")
        V_prime = []  # Output: assigned (service, resource) pairs

        # Feature matrices for cosine similarity
        if len(self.trusted_service_idxs) > 0:
            F_S = self.H[self.trusted_service_idxs]
        else:
            F_S = self.H[self.service_idxs]

        if len(self.trusted_resource_idxs) > 0:
            F_R = self.H[self.trusted_resource_idxs]
        else:
            F_R = self.H[self.resource_idxs]

        used_s, used_r = set(), set()
        for task in workflow:
            task_feat = torch.tensor(task.get("task_features", [0.0] * 16), dtype=torch.float32)
            res_feat = torch.tensor(task.get("resource_features", [0.0] * 16), dtype=torch.float32)

            # find_best_service (exclude already-used for diversity)
            s_idx = find_best_service(self.H, F_S, task_feat, self.trusted_service_idxs, used_s)
            # find_best_resource
            r_idx = find_best_resource(self.H, F_R, res_feat, self.trusted_resource_idxs, used_r)
            if s_idx >= 0:
                used_s.add(s_idx)
            if r_idx >= 0:
                used_r.add(r_idx)

            s_id = self.tkg.index_node.get(s_idx, "UNKNOWN")
            r_id = self.tkg.index_node.get(r_idx, "UNKNOWN")
            s_meta = self.tkg.services.get(s_id, {})
            r_meta = self.tkg.resources.get(r_id, {})

            d = self._proximity_degree(s_idx, r_idx) if s_idx >= 0 and r_idx >= 0 else 0.5

            V_prime.append({
                "task_id": task.get("task_id"),
                "task_name": task.get("task_name"),
                "service_id": s_id,
                "service_name": s_meta.get("name", ""),
                "resource_id": r_id,
                "resource_name": r_meta.get("name", ""),
                "proximity_degree": round(d, 4),
                "qos": s_meta.get("qos", {})
            })

        # Evaluate composition (Eq. score)
        score = self._composition_score(V_prime)

        result = {
            "workflow_size": len(workflow),
            "assignments": V_prime,
            "composition_score": round(score, 4),
            "num_services": len({a["service_id"] for a in V_prime}),
            "num_resources": len({a["resource_id"] for a in V_prime}),
            "trust_summary": {
                "trusted_set_size": len(self.trusted_service_idxs),
                "conflict_set_size": len(self.conflict_set),
                "trust_set_size": len(self.trust_set)
            }
        }
        return result

    def _composition_score(self, assignments: list) -> float:
        """
        Eq. Score(W^p) = d/(1-d) * 1/(|S'|*|R'|) * sum(QoS_i * |S_i in W^p|)
        """
        if not assignments:
            return 0.0
        unique_services = {a["service_id"] for a in assignments}
        unique_resources = {a["resource_id"] for a in assignments}
        S_prime = len(unique_services)
        R_prime = len(unique_resources)

        # Average proximity degree
        d = np.mean([a["proximity_degree"] for a in assignments])
        if d >= 1.0:
            d = 0.99
        conflict_factor = 1 - d
        if conflict_factor < 1e-6:
            conflict_factor = 1e-6

        # QoS sum: use reliability as primary QoS metric
        qos_sum = 0.0
        for a in assignments:
            qos = a.get("qos", {})
            reliability = qos.get("reliability", 0.5)
            count = sum(1 for b in assignments if b["service_id"] == a["service_id"])
            qos_sum += reliability * count

        score = (d / conflict_factor) * (1.0 / max(S_prime * R_prime, 1)) * qos_sum
        return float(score)

    def conflict_severity(self, assignments: list) -> float:
        """
        Compute conflict severity = 1 - average(proximity_degree).
        """
        if not assignments:
            return 1.0
        d = np.mean([a["proximity_degree"] for a in assignments])
        return round(1.0 - d, 4)

    def trust_score(self, assignments: list) -> float:
        """Average proximity (trust) score of the composition."""
        if not assignments:
            return 0.0
        return round(float(np.mean([a["proximity_degree"] for a in assignments])), 4)
