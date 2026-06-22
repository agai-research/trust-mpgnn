"""
main-tkg.py - Main file for Trust Knowledge Graph construction.
Loads dataset, builds TKG, exports graph structure.
Author: H. Mezni
"""

import os
import sys
import json
import argparse
import logging

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.utils import setup_logging, load_json, save_json, load_config
from src.tkg.tkg import TrustKnowledgeGraph

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build Trust Knowledge Graph from dataset")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset path")
    parser.add_argument("--out", type=str, default=None, help="Override TKG output path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log_dir = cfg["paths"].get("logs", "logs")
    setup_logging(log_dir)

    dataset_path = args.dataset or cfg["paths"]["dataset"]
    tkg_out = args.out or cfg["paths"]["tkg_output"]

    logger.info("=== Trust Knowledge Graph Construction ===")
    logger.info(f"Dataset: {dataset_path}")

    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}. Run data/gen_dataset.py first.")
        sys.exit(1)

    # Load dataset and build TKG
    dataset = load_json(dataset_path)
    tkg = TrustKnowledgeGraph()
    tkg.load_from_dataset(dataset)

    # Print TKG statistics
    stats = tkg.stats()
    logger.info("TKG Statistics:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")

    # Export TKG to JSON
    tkg.to_json(tkg_out)
    logger.info(f"TKG structure saved to: {tkg_out}")


if __name__ == "__main__":
    main()
