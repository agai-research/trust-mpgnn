"""
exp_icps_size.py - Experiment: Composition time under increasing ICPS size.
ICPS sizes: 1000, 1500, 2000, 2500, 3000 nodes.
Matches Figure "Composition time under increasing ICPS size" in the paper.
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
from src.composition.baselines import TrustGNN, FFCAIoTSC, TQoSC, compute_metrics

logger = logging.getLogger(__name__)

ICPS_SIZES = [1000, 1500, 2000, 2500, 3000]
WORKFLOW_SIZE = 10


def main():
    cfg = load_config(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json"))
    setup_logging(cfg["paths"].get("logs", "logs"))
    logger.info("=== Experiment: ICPS Size Scalability ===")

    results = {"experiment": "icps_size", "workflow_size": WORKFLOW_SIZE,
               "methods": {m: [] for m in ["Trust-MPGNN", "Trust-GNN", "FFCA-IoTSC", "TQoSC"]}}

    for size in ICPS_SIZES:
        inst_path = f"data/instances/tkg_size_{size}.json"
        if not os.path.exists(inst_path):
            logger.warning(f"Instance not found: {inst_path}. Skipping.")
            continue

        logger.info(f"\n-- ICPS Size: {size} nodes --")
        dataset = load_json(inst_path)

        # Train Trust-MPGNN
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
        conflict_set = {(u, v) for u, s, v in Gamma}
        H_np = H.detach().numpy()
        Gamma_ids = {u for u, _, v in Gamma} | {v for u, _, v in Gamma}

        workflow = generate_workflow(WORKFLOW_SIZE, seed=42)

        # Trust-MPGNN composition time
        t0 = time.time()
        composer = IoTComposer(tkg, H, Delta, Gamma, cfg["composition"])
        result = composer.compose(workflow)
        comp_time = int((time.time() - t0) * 1000)
        m = {"icps_size": size, "composition_time_ms": comp_time}
        results["methods"]["Trust-MPGNN"].append(m)
        logger.info(f"  Trust-MPGNN: {comp_time}ms")

        # Baselines (composition only, using pre-built TKG)
        for bname in ["Trust-GNN", "FFCA-IoTSC", "TQoSC"]:
            t0 = time.time()
            if bname == "Trust-GNN":
                b = TrustGNN(H_np, tkg, Gamma_ids)
            elif bname == "FFCA-IoTSC":
                b = FFCAIoTSC(tkg, conflict_density=0.2)
            else:
                b = TQoSC(tkg)
            b.compose(workflow)
            btime = int((time.time() - t0) * 1000)
            results["methods"][bname].append({"icps_size": size, "composition_time_ms": btime})
            logger.info(f"  {bname}: {btime}ms")

    out_path = "output/exp_icps_size.json"
    save_json(results, out_path)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
