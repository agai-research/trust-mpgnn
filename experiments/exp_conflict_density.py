"""
exp_conflict_density.py - Experiment: Impact of conflict density on composition quality.
Tests conflict densities 10%-50%, fixed ICPS size=2000, workflow=10.
Matches Table 8 and Figure in the paper.
Author: H. Mezni
"""

import os
import sys
import time
import json
import logging
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.utils import setup_logging, load_json, save_json, load_config
from src.tkg.tkg import TrustKnowledgeGraph
from src.tkg.metapaths import METAPATHS
from src.gnn.sampler import MetapathSampler
from src.gnn.model import TrustMPGNN
from src.gnn.trainer import TrustTrainer
from src.composition.composer import IoTComposer
from src.composition.workflow import generate_workflow
from src.composition.baselines import (TrustGNN, FFCAIoTSC, TQoSC, compute_metrics)

logger = logging.getLogger(__name__)

# Paper Table 8 reference values (for validation)
PAPER_VALUES = {
    "Trust-MPGNN": {
        0.1: (98.5, 0.04, 0.95), 0.2: (96.8, 0.07, 0.92),
        0.3: (94.2, 0.11, 0.88), 0.4: (90.1, 0.16, 0.83), 0.5: (86.3, 0.21, 0.78)
    }
}

WORKFLOW_SIZE = 10
DENSITIES = [0.1, 0.2, 0.3, 0.4, 0.5]


def run_trust_mpgnn(dataset: dict, cfg: dict, density: float) -> dict:
    """Run Trust-MPGNN on a dataset instance and return metrics."""
    tkg = TrustKnowledgeGraph()
    tkg.load_from_dataset(dataset)

    sampler = MetapathSampler(tkg, METAPATHS, sampling_size=cfg["gnn"]["sampling_size"])
    neighborhoods = sampler.build_neighborhoods()
    nbr_idx_stack, nbr_mask_stack = sampler.to_padded_tensors(neighborhoods)

    model = TrustMPGNN(16, cfg["gnn"]["embed_dim"], cfg["gnn"]["num_layers"],
                        len(METAPATHS), cfg["gnn"]["dropout"])
    trainer = TrustTrainer(model, tkg, sampler, cfg["gnn"])
    H = trainer.train(nbr_idx_stack, nbr_mask_stack)
    _, Delta, Gamma = trainer.predict_relations(H)

    workflow = generate_workflow(WORKFLOW_SIZE, seed=42)
    t0 = time.time()
    composer = IoTComposer(tkg, H, Delta, Gamma, cfg["composition"])
    result = composer.compose(workflow)
    comp_time = int((time.time() - t0) * 1000)

    conflict_set = {(u, v) for u, score, v in Gamma}
    metrics = compute_metrics(result, conflict_set, tkg)
    metrics["composition_time_ms"] = comp_time
    metrics["density"] = density
    return metrics, tkg, H, Delta, Gamma


def run_baseline(name: str, tkg, H_np: np.ndarray, Gamma: list,
                 density: float, workflow: list) -> dict:
    """Run a baseline method."""
    conflict_set = {(u, v) for u, score, v in Gamma}
    Gamma_ids = {u for u, _, v in Gamma} | {v for u, _, v in Gamma}

    t0 = time.time()
    if name == "Trust-GNN":
        b = TrustGNN(H_np, tkg, Gamma_ids)
        result = b.compose(workflow)
    elif name == "FFCA-IoTSC":
        b = FFCAIoTSC(tkg, conflict_density=density)
        result = b.compose(workflow)
    elif name == "TQoSC":
        b = TQoSC(tkg)
        result = b.compose(workflow)
    else:
        return {}

    comp_time = int((time.time() - t0) * 1000)
    metrics = compute_metrics(result, conflict_set, tkg)
    metrics["composition_time_ms"] = comp_time
    metrics["density"] = density
    return metrics


def main():
    cfg = load_config(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json"))
    setup_logging(cfg["paths"].get("logs", "logs"))
    logger.info("=== Experiment: Impact of Conflict Density ===")

    results = {"experiment": "conflict_density", "workflow_size": WORKFLOW_SIZE,
               "methods": {m: [] for m in ["Trust-MPGNN", "Trust-GNN", "FFCA-IoTSC", "TQoSC"]}}

    for density in DENSITIES:
        inst_path = f"data/instances/tkg_density_{int(density*100)}.json"
        if not os.path.exists(inst_path):
            logger.warning(f"Instance not found: {inst_path}. Skipping density={density}")
            continue

        logger.info(f"\n-- Conflict Density: {int(density*100)}% --")
        dataset = load_json(inst_path)

        # Trust-MPGNN
        metrics, tkg, H, Delta, Gamma = run_trust_mpgnn(dataset, cfg, density)
        results["methods"]["Trust-MPGNN"].append(metrics)
        logger.info(f"  Trust-MPGNN: success={metrics['success_rate']:.3f}, "
                    f"trust={metrics['trust_score']:.3f}, sev={metrics['conflict_severity']:.3f}")

        H_np = H.detach().numpy()
        workflow = generate_workflow(WORKFLOW_SIZE, seed=42)

        # Baselines
        for bname in ["Trust-GNN", "FFCA-IoTSC", "TQoSC"]:
            bm = run_baseline(bname, tkg, H_np, Gamma, density, workflow)
            results["methods"][bname].append(bm)
            logger.info(f"  {bname}: success={bm.get('success_rate',0):.3f}, "
                        f"trust={bm.get('trust_score',0):.3f}, sev={bm.get('conflict_severity',0):.3f}")

    out_path = "output/exp_conflict_density.json"
    save_json(results, out_path)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
