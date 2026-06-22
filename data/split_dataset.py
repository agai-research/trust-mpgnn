"""
split_dataset.py - Generate dataset instances for different experimental configurations.
Author: H. Mezni
"""

import json
import os
import random
import logging
import argparse
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_dataset(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def sample_instance(dataset: dict, num_nodes: int, conflict_density: float, seed: int = 42) -> dict:
    """
    Create a TKG instance with given node count and conflict density.
    num_nodes = total providers + services + resources.
    """
    random.seed(seed)
    providers = dataset["providers"]
    services = dataset["services"]
    resources = dataset["resources"]
    relations = dataset["relations"]

    # Proportional sampling
    ratio_p = len(providers) / (len(providers) + len(services) + len(resources))
    ratio_s = len(services) / (len(providers) + len(services) + len(resources))
    ratio_r = len(resources) / (len(providers) + len(services) + len(resources))

    n_p = max(1, int(num_nodes * ratio_p))
    n_s = max(1, int(num_nodes * ratio_s))
    n_r = max(1, int(num_nodes * ratio_r))
    n_p = min(n_p, len(providers))
    n_s = min(n_s, len(services))
    n_r = min(n_r, len(resources))

    sampled_providers = random.sample(providers, n_p)
    sampled_services = random.sample(services, n_s)
    sampled_resources = random.sample(resources, n_r)

    p_ids = {p["id"] for p in sampled_providers}
    s_ids = {s["id"] for s in sampled_services}
    r_ids = {r["id"] for r in sampled_resources}
    all_ids = p_ids | s_ids | r_ids

    # Filter relations to sampled entities
    valid_rels = [r for r in relations if r["head"] in all_ids and r["tail"] in all_ids]

    # Adjust conflict density: re-tag oppose/conflict edges
    non_conflict_types = {"TRUST", "SUPPORT", "ALLIED"}
    conflict_types = {"OPPOSE", "CONFLICT"}
    adjusted = []
    for rel in valid_rels:
        r = dict(rel)
        if rel["type"] in non_conflict_types:
            if random.random() < conflict_density:
                # Flip to conflict
                if rel["head_type"] == "Service" and rel["tail_type"] == "Resource":
                    r["relation"] = r["type"] = "OPPOSE"
                elif rel["head_type"] == "Service" and rel["tail_type"] == "Service":
                    r["relation"] = r["type"] = "CONFLICT"
                elif rel["head_type"] == "Resource" and rel["tail_type"] == "Resource":
                    r["relation"] = r["type"] = "CONFLICT"
        adjusted.append(r)

    from collections import Counter
    rcnt = Counter(r["type"] for r in adjusted)

    instance = {
        "metadata": {
            "num_nodes": n_p + n_s + n_r,
            "num_providers": n_p,
            "num_services": n_s,
            "num_resources": n_r,
            "num_relations": len(adjusted),
            "conflict_density": conflict_density,
            "seed": seed,
            "relation_counts": dict(rcnt)
        },
        "providers": sampled_providers,
        "services": sampled_services,
        "resources": sampled_resources,
        "relations": adjusted
    }
    return instance


def main():
    parser = argparse.ArgumentParser(description="Create dataset instances for experiments")
    parser.add_argument("--dataset", type=str, default="data/raw/dataset.json")
    parser.add_argument("--out_dir", type=str, default="data/instances")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    os.makedirs(args.out_dir, exist_ok=True)

    # Instances for ICPS size experiment: 1000, 1500, 2000, 2500, 3000 nodes
    for size in [1000, 1500, 2000, 2500, 3000]:
        inst = sample_instance(dataset, size, conflict_density=0.2, seed=args.seed)
        path = os.path.join(args.out_dir, f"tkg_size_{size}.json")
        with open(path, "w") as f:
            json.dump(inst, f, indent=2)
        logger.info(f"Saved instance size={size} -> {path}")

    # Instances for conflict density experiment: fixed 2000 nodes, density 10%-50%
    for density in [0.1, 0.2, 0.3, 0.4, 0.5]:
        inst = sample_instance(dataset, 2000, conflict_density=density, seed=args.seed)
        path = os.path.join(args.out_dir, f"tkg_density_{int(density*100)}.json")
        with open(path, "w") as f:
            json.dump(inst, f, indent=2)
        logger.info(f"Saved instance density={density} -> {path}")

    # Instance for trust threshold experiment: 2000 nodes, 20% conflict
    inst = sample_instance(dataset, 2000, conflict_density=0.2, seed=args.seed)
    path = os.path.join(args.out_dir, "tkg_threshold.json")
    with open(path, "w") as f:
        json.dump(inst, f, indent=2)
    logger.info(f"Saved threshold instance -> {path}")

    # Instance for workflow complexity: 2000 nodes, 20% conflict
    inst = sample_instance(dataset, 2000, conflict_density=0.2, seed=args.seed)
    path = os.path.join(args.out_dir, "tkg_workflow.json")
    with open(path, "w") as f:
        json.dump(inst, f, indent=2)
    logger.info(f"Saved workflow complexity instance -> {path}")

    # First-test small instance: 200 nodes
    inst_small = sample_instance(dataset, 200, conflict_density=0.2, seed=args.seed)
    path = os.path.join(args.out_dir, "tkg_test.json")
    with open(path, "w") as f:
        json.dump(inst_small, f, indent=2)
    logger.info(f"Saved test instance -> {path}")

    logger.info("All dataset instances created.")


if __name__ == "__main__":
    main()
