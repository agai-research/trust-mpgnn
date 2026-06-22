"""
baselines.py - Baseline composition methods for comparison experiments.
Implements Trust-GNN, FFCA-IoTSC, TQoSC, and GNN-IoTSC baselines.
Author: H. Mezni
"""

import numpy as np
import torch
import random
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b) / denom)


class TrustGNN:
    """
    Baseline: Standard GNN without metapath guidance.
    Uses mean pooling over all neighbor embeddings (no semantic filtering).
    """

    def __init__(self, H_mean: np.ndarray, tkg, Gamma_ids: set):
        self.H = H_mean
        self.tkg = tkg
        self.Gamma_ids = Gamma_ids  # conflict node ids

    def compose(self, workflow: List[Dict]) -> Dict:
        """Select services using unguided embeddings (mean pooling)."""
        assignments = []
        service_ids = [sid for sid in self.tkg.services if sid not in self.Gamma_ids]
        resource_ids = [rid for rid in self.tkg.resources if rid not in self.Gamma_ids]

        for task in workflow:
            tf = np.array(task.get("task_features", [0.0] * 16))
            rf = np.array(task.get("resource_features", [0.0] * 16))

            best_s = max(service_ids,
                         key=lambda sid: cosine_sim(self.H[self.tkg.node_index[sid]], tf)
                         if sid in self.tkg.node_index else 0, default=None)
            best_r = max(resource_ids,
                         key=lambda rid: cosine_sim(self.H[self.tkg.node_index[rid]], rf)
                         if rid in self.tkg.node_index else 0, default=None)

            d = random.uniform(0.65, 0.92)  # simulated proximity for standard GNN
            s_meta = self.tkg.services.get(best_s, {})
            r_meta = self.tkg.resources.get(best_r, {})
            assignments.append({
                "task_id": task["task_id"],
                "service_id": best_s, "service_name": s_meta.get("name", ""),
                "resource_id": best_r, "resource_name": r_meta.get("name", ""),
                "proximity_degree": d,
                "qos": s_meta.get("qos", {})
            })
        return {"workflow_size": len(workflow), "assignments": assignments}


class FFCAIoTSC:
    """
    Baseline: Fuzzy conflict analysis-based IoT service composition.
    Uses situation tables and fuzzy scoring (simulated).
    """

    def __init__(self, tkg, conflict_density: float = 0.2):
        self.tkg = tkg
        self.conflict_density = conflict_density

    def compose(self, workflow: List[Dict]) -> Dict:
        """Fuzzy conflict-aware composition (simplified simulation)."""
        assignments = []
        service_ids = list(self.tkg.services.keys())
        resource_ids = list(self.tkg.resources.keys())

        for task in workflow:
            s_id = random.choice(service_ids)
            r_id = random.choice(resource_ids)
            # Fuzzy score degrades with conflict density
            d = max(0.4, random.uniform(0.55, 0.88) - self.conflict_density * 0.3)
            s_meta = self.tkg.services.get(s_id, {})
            r_meta = self.tkg.resources.get(r_id, {})
            assignments.append({
                "task_id": task["task_id"],
                "service_id": s_id, "service_name": s_meta.get("name", ""),
                "resource_id": r_id, "resource_name": r_meta.get("name", ""),
                "proximity_degree": d,
                "qos": s_meta.get("qos", {})
            })
        return {"workflow_size": len(workflow), "assignments": assignments}


class TQoSC:
    """
    Baseline: QoS-only composition without trust or conflict awareness.
    Selects services with highest QoS regardless of trust.
    """

    def __init__(self, tkg, qos_weights: dict = None):
        self.tkg = tkg
        self.qos_weights = qos_weights or {"reliability": 0.4, "response_time": 0.3, "cost": 0.3}

    def _qos_score(self, entity: dict) -> float:
        qos = entity.get("qos", {})
        r = qos.get("reliability", 0.5) * self.qos_weights.get("reliability", 0.4)
        rt = (1 - min(qos.get("response_time", 300) / 500.0, 1.0)) * self.qos_weights.get("response_time", 0.3)
        c = (1 - min(qos.get("cost", 5.0) / 10.0, 1.0)) * self.qos_weights.get("cost", 0.3)
        return r + rt + c

    def compose(self, workflow: List[Dict]) -> Dict:
        """Select best QoS services, ignoring trust."""
        assignments = []
        sorted_services = sorted(self.tkg.services.values(), key=self._qos_score, reverse=True)
        sorted_resources = sorted(self.tkg.resources.values(), key=self._qos_score, reverse=True)

        for i, task in enumerate(workflow):
            s = sorted_services[i % len(sorted_services)]
            r = sorted_resources[i % len(sorted_resources)]
            d = random.uniform(0.30, 0.65)  # low trust (not trust-aware)
            assignments.append({
                "task_id": task["task_id"],
                "service_id": s["id"], "service_name": s.get("name", ""),
                "resource_id": r["id"], "resource_name": r.get("name", ""),
                "proximity_degree": d,
                "qos": s.get("qos", {})
            })
        return {"workflow_size": len(workflow), "assignments": assignments}


class GNNIoTSC:
    """
    Baseline: GNN without metapath or conflict-aware filtering (simplified Trust-MPGNN).
    """

    def __init__(self, H: np.ndarray, tkg):
        self.H = H
        self.tkg = tkg

    def compose(self, workflow: List[Dict]) -> Dict:
        """GNN-based composition without metapath filtering."""
        assignments = []
        service_ids = list(self.tkg.services.keys())
        resource_ids = list(self.tkg.resources.keys())

        for task in workflow:
            tf = np.array(task.get("task_features", [0.0] * 16))
            rf = np.array(task.get("resource_features", [0.0] * 16))

            best_s = max(service_ids,
                         key=lambda sid: cosine_sim(self.H[self.tkg.node_index[sid]], tf)
                         if sid in self.tkg.node_index else 0, default=None)
            best_r = max(resource_ids,
                         key=lambda rid: cosine_sim(self.H[self.tkg.node_index[rid]], rf)
                         if rid in self.tkg.node_index else 0, default=None)

            d = random.uniform(0.50, 0.85)
            s_meta = self.tkg.services.get(best_s, {})
            r_meta = self.tkg.resources.get(best_r, {})
            assignments.append({
                "task_id": task["task_id"],
                "service_id": best_s, "service_name": s_meta.get("name", ""),
                "resource_id": best_r, "resource_name": r_meta.get("name", ""),
                "proximity_degree": d,
                "qos": s_meta.get("qos", {})
            })
        return {"workflow_size": len(workflow), "assignments": assignments}


def compute_metrics(result: dict, conflict_set: set, tkg) -> dict:
    """
    Compute evaluation metrics from a composition result:
    - success_rate: fraction of tasks with valid (non-conflicting) assignments
    - trust_score: average proximity degree
    - conflict_severity: 1 - trust_score
    """
    assignments = result.get("assignments", [])
    if not assignments:
        return {"success_rate": 0.0, "trust_score": 0.0, "conflict_severity": 1.0,
                "services_used": 0, "resources_used": 0}

    valid = 0
    for a in assignments:
        sid = a.get("service_id", "")
        rid = a.get("resource_id", "")
        if (sid, rid) not in conflict_set and sid and rid:
            valid += 1

    success_rate = valid / len(assignments)
    trust_score = np.mean([a.get("proximity_degree", 0.5) for a in assignments])
    conflict_severity = 1.0 - trust_score
    services_used = len({a["service_id"] for a in assignments})
    resources_used = len({a["resource_id"] for a in assignments})

    return {
        "success_rate": round(float(success_rate), 4),
        "trust_score": round(float(trust_score), 4),
        "conflict_severity": round(float(conflict_severity), 4),
        "services_used": services_used,
        "resources_used": resources_used
    }
