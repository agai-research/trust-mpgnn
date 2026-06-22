"""
exp_time_density.py - Experiment: Composition time under varying conflict density.
Matches Table 11 in the paper.
Author: H. Mezni
"""

import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.utils import setup_logging, load_json, save_json, load_config
from src.tkg.tkg import TrustKnowledgeGraph
from src.tkg.metapaths import METAPATHS
from src.gnn.sampler import MetapathSampler
from src.gnn.model import TrustMPGNN
from src.gnn.trainer import TrustTrainer
from src.composition.composer import IoTComposer
from src.composition.workflow import generate_workflow
from src.composition.baselines import TrustGNN, FFCAIoTSC, TQoSC

logger = logging.getLogger(__name__)

DENSITIES = [0.1, 0.2, 0.3, 0.4, 0.5]
WORKFLOW_SIZE = 10


def main():
    cfg = load_config(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json"))
    setup_logging(cfg["paths"].get("logs", "logs"))
    logger.info("=== Experiment: Composition Time under Varying Conflict Density ===")

    results = {"experiment": "time_vs_density", "workflow_size": WORKFLOW_SIZE,
               "methods": {m: [] for m in ["Trust-MPGNN", "Trust-GNN", "FFCA-IoTSC", "TQoSC"]}}

    for density in DENSITIES:
        inst_path = f"data/instances/tkg_density_{int(density*100)}.json"
        if not os.path.exists(inst_path):
            logger.warning(f"Missing: {inst_path}")
            continue

        logger.info(f"\n-- Conflict Density: {int(density*100)}% --")
        dataset = load_json(inst_path)

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
        H_np = H.detach().numpy()
        Gamma_ids = {u for u, _, v in Gamma} | {v for u, _, v in Gamma}
        workflow = generate_workflow(WORKFLOW_SIZE, seed=42)

        # Trust-MPGNN
        t0 = time.time()
        IoTComposer(tkg, H, Delta, Gamma, cfg["composition"]).compose(workflow)
        tm = int((time.time() - t0) * 1000)
        results["methods"]["Trust-MPGNN"].append({"density": density, "time_ms": tm})
        logger.info(f"  Trust-MPGNN: {tm}ms")

        for bname in ["Trust-GNN", "FFCA-IoTSC", "TQoSC"]:
            t0 = time.time()
            if bname == "Trust-GNN":
                TrustGNN(H_np, tkg, Gamma_ids).compose(workflow)
            elif bname == "FFCA-IoTSC":
                FFCAIoTSC(tkg, conflict_density=density).compose(workflow)
            else:
                TQoSC(tkg).compose(workflow)
            bt = int((time.time() - t0) * 1000)
            results["methods"][bname].append({"density": density, "time_ms": bt})
            logger.info(f"  {bname}: {bt}ms")

    out_path = "output/exp_time_density.json"
    save_json(results, out_path)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
