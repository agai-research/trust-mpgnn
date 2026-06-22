"""
exp_threshold.py - Experiment: Impact of trust threshold theta on composition quality.
Theta varies from 0.5 to 0.8. Fixed ICPS=2000, density=20%, workflow=10.
Matches Table 10 in the paper.
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
from src.composition.baselines import compute_metrics

logger = logging.getLogger(__name__)

THRESHOLDS = [0.5, 0.6, 0.7, 0.8]
WORKFLOW_SIZE = 10


def main():
    cfg = load_config(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json"))
    setup_logging(cfg["paths"].get("logs", "logs"))
    logger.info("=== Experiment: Trust Threshold Sensitivity ===")

    inst_path = "data/instances/tkg_threshold.json"
    if not os.path.exists(inst_path):
        logger.error(f"Instance not found: {inst_path}. Run main-exp.py first.")
        sys.exit(1)

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

    results = {"experiment": "trust_threshold", "workflow_size": WORKFLOW_SIZE,
               "results": []}

    workflow = generate_workflow(WORKFLOW_SIZE, seed=42)

    for theta in THRESHOLDS:
        logger.info(f"\n-- Threshold theta={theta} --")
        # Re-predict with different theta
        trainer.theta = theta
        E_hat, Delta, Gamma = trainer.predict_relations(H)
        conflict_set = {(u, v) for u, s, v in Gamma}

        composer = IoTComposer(tkg, H, Delta, Gamma, cfg["composition"])
        result = composer.compose(workflow)
        metrics = compute_metrics(result, conflict_set, tkg)

        row = {
            "theta": theta,
            "success_rate": metrics["success_rate"],
            "trust_score": metrics["trust_score"],
            "conflict_severity": metrics["conflict_severity"],
            "delta_size": len(Delta),
            "gamma_size": len(Gamma)
        }
        results["results"].append(row)
        logger.info(f"  theta={theta}: success={row['success_rate']:.3f}, "
                    f"trust={row['trust_score']:.3f}, sev={row['conflict_severity']:.3f}, "
                    f"|Delta|={row['delta_size']}, |Gamma|={row['gamma_size']}")

    out_path = "output/exp_threshold.json"
    save_json(results, out_path)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
