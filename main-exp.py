"""
main-exp.py - Generate all dataset instances needed for experiments.
Author: H. Mezni
"""

import os
import sys
import argparse
import logging
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.utils import setup_logging, load_config

logger = logging.getLogger(__name__)


def run(cmd: list):
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed: {' '.join(cmd)}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Generate experiment datasets and instances")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg["paths"].get("logs", "logs"))

    logger.info("=== Generating main dataset ===")
    ds_cfg = cfg["tkg"]
    run([sys.executable, "data/gen_dataset.py",
         "--providers", str(ds_cfg["num_providers"]),
         "--services", str(ds_cfg["num_services"]),
         "--resources", str(ds_cfg["num_resources"]),
         "--relations", str(ds_cfg["num_relations"]),
         "--conflict_density", "0.2",
         "--seed", str(args.seed),
         "--out", cfg["paths"]["dataset"]])

    logger.info("=== Splitting dataset into experiment instances ===")
    run([sys.executable, "data/split_dataset.py",
         "--dataset", cfg["paths"]["dataset"],
         "--out_dir", "data/instances",
         "--seed", str(args.seed)])

    logger.info("=== All experiment instances ready ===")


if __name__ == "__main__":
    main()
