"""
trust-mpgnn.py - Main prototype runner for Trust-MPGNN.
Runs the full pipeline: TKG -> Sampling -> Embedding -> Prediction -> Composition.
Author: H. Mezni
"""

import os
import sys
import json
import time
import argparse
import logging
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.utils import setup_logging, load_json, save_json, load_config
from src.tkg.tkg import TrustKnowledgeGraph
from src.tkg.metapaths import METAPATHS
from src.gnn.sampler import MetapathSampler
from src.gnn.model import TrustMPGNN
from src.gnn.trainer import TrustTrainer
from src.composition.composer import IoTComposer
from src.composition.workflow import generate_workflow, SAMPLE_QUERIES

logger = logging.getLogger(__name__)


def run_pipeline(cfg: dict, dataset_path: str, workflow_size: int = 10,
                 trust_threshold: float = None):
    """
    Full Trust-MPGNN pipeline:
    1. TKG construction
    2. Metapath-guided sampling
    3. Trust embedding (GNN training)
    4. Trust/conflict prediction
    5. IoT service composition
    """
    t0 = time.time()
    gnn_cfg = cfg["gnn"]
    if trust_threshold is not None:
        gnn_cfg["trust_threshold"] = trust_threshold

    # -- Step 1: TKG Construction --
    logger.info("--- Step 1: TKG Construction ---")
    dataset = load_json(dataset_path)
    tkg = TrustKnowledgeGraph()
    tkg.load_from_dataset(dataset)
    tkg.to_json(cfg["paths"]["tkg_output"])
    stats = tkg.stats()
    logger.info(f"TKG: {stats}")

    # -- Step 2: Metapath-guided Sampling --
    logger.info("--- Step 2: Metapath-guided Sampling ---")
    sampler = MetapathSampler(tkg, METAPATHS, sampling_size=gnn_cfg["sampling_size"])
    neighborhoods = sampler.build_neighborhoods()
    nbr_idx_stack, nbr_mask_stack = sampler.to_padded_tensors(neighborhoods)

    # -- Step 3: Trust Embedding (Algorithm 1) --
    logger.info("--- Step 3: Trust Embedding ---")
    model = TrustMPGNN(
        input_dim=16,
        embed_dim=gnn_cfg["embed_dim"],
        num_layers=gnn_cfg["num_layers"],
        num_metapaths=len(METAPATHS),
        dropout=gnn_cfg["dropout"]
    )
    trainer = TrustTrainer(model, tkg, sampler, gnn_cfg)
    H = trainer.train(nbr_idx_stack, nbr_mask_stack)
    trainer.save_embeddings(H, cfg["paths"]["embeddings"])

    # -- Step 4: Trust Prediction --
    logger.info("--- Step 4: Trust/Conflict Prediction ---")
    E_hat, Delta, Gamma = trainer.predict_relations(H)

    # -- Step 5: Composition (Algorithm 2) --
    logger.info(f"--- Step 5: IoT Service Composition (workflow={workflow_size}) ---")
    workflow = generate_workflow(workflow_size, seed=42)
    composer = IoTComposer(tkg, H, Delta, Gamma, cfg["composition"])
    result = composer.compose(workflow)
    result["trust_score"] = composer.trust_score(result["assignments"])
    result["conflict_severity"] = composer.conflict_severity(result["assignments"])
    result["elapsed_sec"] = round(time.time() - t0, 2)

    logger.info(f"Composition complete: score={result['composition_score']}, "
                f"trust={result['trust_score']}, severity={result['conflict_severity']}")

    return result, E_hat, Delta, Gamma, H, tkg


def main():
    parser = argparse.ArgumentParser(description="Run Trust-MPGNN prototype")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--workflow", type=int, default=10, help="Workflow size")
    parser.add_argument("--theta", type=float, default=None, help="Trust threshold")
    parser.add_argument("--query", type=int, default=None, help="Use predefined query (1-4)")
    parser.add_argument("--out", type=str, default="output/composition_result.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg["paths"].get("logs", "logs"))

    dataset_path = args.dataset or cfg["paths"]["dataset"]
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}. Run: python main-exp.py")
        sys.exit(1)

    logger.info("=== Trust-MPGNN: Conflict-aware IoT Service Composition ===")
    logger.info(f"Author: {cfg.get('author', 'H. Mezni')}")

    # Choose workflow
    if args.query and 1 <= args.query <= len(SAMPLE_QUERIES):
        q = SAMPLE_QUERIES[args.query - 1]
        logger.info(f"Using predefined query Q{args.query}: {q['description']}")
        workflow_size = len(q["workflow"])
    else:
        workflow_size = args.workflow

    result, E_hat, Delta, Gamma, H, tkg = run_pipeline(
        cfg, dataset_path, workflow_size, args.theta
    )

    # Output
    output = {
        "pipeline": "Trust-MPGNN",
        "author": "H. Mezni",
        "dataset": dataset_path,
        "workflow_size": workflow_size,
        "trust_threshold": cfg["gnn"]["trust_threshold"],
        "tkg_stats": tkg.stats(),
        "trust_relations": len(Delta),
        "conflict_relations": len(Gamma),
        "composition": result
    }

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    save_json(output, args.out)
    logger.info(f"\n{'='*50}")
    logger.info("COMPOSITION RESULT:")
    logger.info(f"  Trust Score:       {result['trust_score']}")
    logger.info(f"  Conflict Severity: {result['conflict_severity']}")
    logger.info(f"  Composition Score: {result['composition_score']}")
    logger.info(f"  Services Used:     {result['num_services']}")
    logger.info(f"  Resources Used:    {result['num_resources']}")
    logger.info(f"  Elapsed:           {result['elapsed_sec']}s")
    logger.info(f"  Results saved:     {args.out}")
    logger.info(f"{'='*50}")

    # Print assignments
    logger.info("\nTask Assignments:")
    for a in result["assignments"]:
        logger.info(f"  [{a['task_id']}] {a.get('task_name','')} -> "
                    f"Service: {a['service_name']} | Resource: {a['resource_name']} | "
                    f"Trust: {a['proximity_degree']}")


if __name__ == "__main__":
    main()
