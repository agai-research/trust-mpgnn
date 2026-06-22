"""
test_prototype.py - First test: validate Trust-MPGNN prototype on small TKG + sample queries.
Author: H. Mezni
"""

import os
import sys
import logging
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.utils import setup_logging, load_json, save_json, load_config
from src.tkg.tkg import TrustKnowledgeGraph
from src.tkg.metapaths import METAPATHS
from src.gnn.sampler import MetapathSampler
from src.gnn.model import TrustMPGNN
from src.gnn.trainer import TrustTrainer
from src.composition.composer import IoTComposer
from src.composition.workflow import SAMPLE_QUERIES
from src.composition.baselines import compute_metrics

logger = logging.getLogger(__name__)


def main():
    cfg = load_config(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json"))
    # Use fewer epochs for quick test
    cfg["gnn"]["epochs"] = 30
    cfg["gnn"]["sampling_size"] = 10
    setup_logging(cfg["paths"].get("logs", "logs"))

    logger.info("=== First Test: Trust-MPGNN Prototype Validation ===")
    logger.info("Author: H. Mezni")

    inst_path = "data/instances/tkg_test.json"
    if not os.path.exists(inst_path):
        logger.error("Test instance not found. Run: python main-exp.py")
        sys.exit(1)

    dataset = load_json(inst_path)
    logger.info(f"Test TKG: {dataset['metadata']['num_nodes']} nodes, "
                f"{dataset['metadata']['num_relations']} relations")

    # Build TKG
    tkg = TrustKnowledgeGraph()
    tkg.load_from_dataset(dataset)
    stats = tkg.stats()
    logger.info(f"TKG stats: {stats}")

    # Sample + train
    sampler = MetapathSampler(tkg, METAPATHS, sampling_size=cfg["gnn"]["sampling_size"])
    neighborhoods = sampler.build_neighborhoods()
    nbr_idx_stack, nbr_mask_stack = sampler.to_padded_tensors(neighborhoods)

    model = TrustMPGNN(16, cfg["gnn"]["embed_dim"], 2, len(METAPATHS), cfg["gnn"]["dropout"])
    trainer = TrustTrainer(model, tkg, sampler, cfg["gnn"])
    H = trainer.train(nbr_idx_stack, nbr_mask_stack)
    _, Delta, Gamma = trainer.predict_relations(H)
    conflict_set = {(u, v) for u, s, v in Gamma}

    logger.info(f"\nTrust relations (Delta): {len(Delta)}")
    logger.info(f"Conflict relations (Gamma): {len(Gamma)}")

    # Test all sample queries
    all_results = []
    for q in SAMPLE_QUERIES:
        logger.info(f"\n--- Query {q['query_id']}: {q['description']} ---")
        workflow = q["workflow"]
        composer = IoTComposer(tkg, H, Delta, Gamma, cfg["composition"])
        result = composer.compose(workflow)
        metrics = compute_metrics(result, conflict_set, tkg)

        logger.info(f"  Workflow size: {len(workflow)}")
        logger.info(f"  Trust score:       {metrics['trust_score']:.3f}")
        logger.info(f"  Conflict severity: {metrics['conflict_severity']:.3f}")
        logger.info(f"  Success rate:      {metrics['success_rate']:.3f}")
        logger.info(f"  Composition score: {result['composition_score']:.4f}")
        logger.info("  Assignments:")
        for a in result["assignments"]:
            logger.info(f"    [{a['task_id']}] {a.get('task_name','')} -> "
                        f"Service: {a['service_name'][:30]:30s} | "
                        f"Resource: {a['resource_name'][:30]:30s} | Trust={a['proximity_degree']:.3f}")

        all_results.append({
            "query_id": q["query_id"],
            "description": q["description"],
            "metrics": metrics,
            "composition_score": result["composition_score"],
            "assignments": result["assignments"]
        })

    out_path = "output/test_results.json"
    save_json({"test": "prototype_validation", "results": all_results}, out_path)
    logger.info(f"\nAll test results saved to {out_path}")
    logger.info("=== Test Complete ===")


if __name__ == "__main__":
    main()
