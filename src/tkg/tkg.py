"""
tkg.py - Trust Knowledge Graph construction using NetworkX.
Loads dataset and builds TKG as a directed heterogeneous graph.
Author: H. Mezni
"""

import json
import os
import logging
import networkx as nx
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

# Valid relation types as defined in the paper
RELATION_TYPES = ["TRUST", "SUPPORT", "OPPOSE", "NEUTRAL", "ALLIED", "CONFLICT"]


class TrustKnowledgeGraph:
    """
    TKG = G = <V, E, T, phi>
    V = Providers U Services U Resources U Features
    T = {TRUST, SUPPORT, OPPOSE, NEUTRAL, ALLIED, CONFLICT}
    """

    def __init__(self):
        self.G = nx.DiGraph()
        self.providers = {}   # id -> metadata
        self.services = {}
        self.resources = {}
        self.node_index = {}  # id -> int index
        self.index_node = {}  # int -> id
        self.relation_index = {r: i for i, r in enumerate(RELATION_TYPES)}

    def load_from_dataset(self, dataset: dict):
        """Populate TKG from a dataset dict (providers, services, resources, relations)."""
        logger.info("Building Trust Knowledge Graph ...")
        self.providers = {p["id"]: p for p in dataset["providers"]}
        self.services = {s["id"]: s for s in dataset["services"]}
        self.resources = {r["id"]: r for r in dataset["resources"]}

        # Add nodes
        idx = 0
        for p in dataset["providers"]:
            self.G.add_node(p["id"], ntype="Provider", **{k: v for k, v in p.items() if k not in ("feature_vector",)})
            self.node_index[p["id"]] = idx; self.index_node[idx] = p["id"]; idx += 1
        for s in dataset["services"]:
            self.G.add_node(s["id"], ntype="Service", **{k: v for k, v in s.items() if k not in ("feature_vector",)})
            self.node_index[s["id"]] = idx; self.index_node[idx] = s["id"]; idx += 1
        for r in dataset["resources"]:
            self.G.add_node(r["id"], ntype="Resource", **{k: v for k, v in r.items() if k not in ("feature_vector",)})
            self.node_index[r["id"]] = idx; self.index_node[idx] = r["id"]; idx += 1

        # Add edges
        added = 0
        for rel in dataset["relations"]:
            h, t, rtype = rel["head"], rel["tail"], rel["type"]
            if h in self.G and t in self.G and rtype in RELATION_TYPES:
                self.G.add_edge(h, t, relation=rtype, rtype_idx=self.relation_index[rtype])
                added += 1
        logger.info(f"TKG built: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges.")
        return self

    def load_from_file(self, path: str):
        """Load dataset from JSON file and build TKG."""
        with open(path) as f:
            dataset = json.load(f)
        return self.load_from_dataset(dataset)

    def get_neighbors_by_relation(self, node_id: str, relation: str) -> list:
        """Return neighbors of node_id connected via given relation type."""
        return [v for u, v, data in self.G.out_edges(node_id, data=True)
                if data.get("relation") == relation]

    def get_metapath_neighbors(self, node_id: str, metapath: list) -> list:
        """
        Follow a metapath (list of relation types) from node_id.
        Returns reachable end nodes.
        """
        current = {node_id}
        for rel in metapath:
            next_nodes = set()
            for n in current:
                next_nodes.update(self.get_neighbors_by_relation(n, rel))
            current = next_nodes
        current.discard(node_id)
        return list(current)

    def node_features(self, node_id: str, feature_dim: int = 16) -> np.ndarray:
        """Extract feature vector for a node."""
        ntype = self.G.nodes[node_id].get("ntype", "")
        src = (self.providers if ntype == "Provider" else
               self.services if ntype == "Service" else self.resources)
        entity = src.get(node_id, {})
        fv = entity.get("feature_vector", None)
        if fv:
            return np.array(fv[:feature_dim], dtype=np.float32)
        # Fallback: zeros
        return np.zeros(feature_dim, dtype=np.float32)

    def stats(self) -> dict:
        """Return TKG statistics."""
        from collections import Counter
        edge_types = Counter(data["relation"] for _, _, data in self.G.edges(data=True))
        node_types = Counter(data.get("ntype", "?") for _, data in self.G.nodes(data=True))
        return {
            "num_nodes": self.G.number_of_nodes(),
            "num_edges": self.G.number_of_edges(),
            "node_types": dict(node_types),
            "edge_types": dict(edge_types)
        }

    def to_json(self, path: str):
        """Export TKG structure to JSON file."""
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        data = {
            "nodes": [{"id": n, **{k: v for k, v in d.items()}}
                      for n, d in self.G.nodes(data=True)],
            "edges": [{"head": u, "tail": v, "relation": d["relation"]}
                      for u, v, d in self.G.edges(data=True)],
            "stats": self.stats()
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"TKG exported to {path}")
