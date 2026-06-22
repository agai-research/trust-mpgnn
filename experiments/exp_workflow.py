"""
exp_workflow.py - Experiment: Impact of workflow complexity on composition quality.
Workflow sizes: 5, 10, 20, 30, 50. Fixed ICPS=2000, density=20%.
Matches Table 9 in the paper.
Author: H. Mezni
"""

import os
import sys
import time
import logging
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
from src.composition.baselines import TrustGNN, FFCAIoTSC, TQoSC, compute_metrics

logger = logging.getLogger(__name__)

WORKFLOW_SIZES = [5, 10, 20, 30, 50]


def main():
    cfg = load_config(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json"))
    setup_logging(cfg["paths"].get("logs", "logs"))
    logger.info("=== Experiment: Workflow Complexity Impact ===")

    inst_path = "data/instances/tkg_workflow.json"
    if not os.path.exists(inst_path):
        logger.error(f"Instance not found: {inst_path}. Run main-exp.py first.")
        sys.exit(1)

    dataset = load_json(inst_path)
    # Train Trust-MPGNN once, test all workflow sizes
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
    conflict_set = {(u, v) for u, score, v in Gamma}
    Gamma_ids = {u for u, _, v in Gamma} | {v for u, _, v in Gamma}

    results = {"experiment": "workflow_complexity", "icps_size": dataset["metadata"]["num_nodes"],
               "conflict_density": 0.2,
               "methods": {m: [] for m in ["Trust-MPGNN", "Trust-GNN", "FFCA-IoTSC", "TQoSC"]}}

    for wsize in WORKFLOW_SIZES:
        logger.info(f"\n-- Workflow Size: {wsize} --")
        workflow = generate_workflow(wsize, seed=42)

        # Trust-MPGNN
        t0 = time.time()
        composer = IoTComposer(tkg, H, Delta, Gamma, cfg["composition"])
        result = composer.compose(workflow)
        comp_time = int((time.time() - t0) * 1000)
        m = compute_metrics(result, conflict_set, tkg)
        m["workflow_size"] = wsize
        m["composition_time_ms"] = comp_time
        results["methods"]["Trust-MPGNN"].append(m)
        logger.info(f"  Trust-MPGNN: success={m['success_rate']:.3f}, trust={m['trust_score']:.3f}, time={comp_time}ms")

        # Baselines
        for bname, BClass in [("Trust-GNN", TrustGNN), ("TQoSC", TQoSC)]:
            t0 = time.time()
            if bname == "Trust-GNN":
                b = TrustGNN(H_np, tkg, Gamma_ids)
            else:
                b = TQoSC(tkg)
            bresult = b.compose(workflow)
            btime = int((time.time() - t0) * 1000)
            bm = compute_metrics(bresult, conflict_set, tkg)
            bm["workflow_size"] = wsize
            bm["composition_time_ms"] = btime
            results["methods"][bname].append(bm)
            logger.info(f"  {bname}: success={bm['success_rate']:.3f}, trust={bm['trust_score']:.3f}, time={btime}ms")

        # FFCA-IoTSC
        t0 = time.time()
        b = FFCAIoTSC(tkg, conflict_density=0.2)
        bresult = b.compose(workflow)
        btime = int((time.time() - t0) * 1000)
        bm = compute_metrics(bresult, conflict_set, tkg)
        bm["workflow_size"] = wsize
        bm["composition_time_ms"] = btime
        results["methods"]["FFCA-IoTSC"].append(bm)
        logger.info(f"  FFCA-IoTSC: success={bm['success_rate']:.3f}, time={btime}ms")

    out_path = "output/exp_workflow.json"
    save_json(results, out_path)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
