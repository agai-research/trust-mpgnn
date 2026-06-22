"""
sampler.py - Metapath-guided neighborhood sampling for Trust-MPGNN.
Implements the Sample(v, m, G) procedure from Algorithm 1.

FIX: added `to_padded_tensors`, which turns the per-node neighbor-id
neighborhoods into a single (num_metapaths, N, K) index tensor + boolean
mask. This is what lets the GNN layers run as vectorized tensor ops
instead of a Python loop over every node (see src/gnn/model.py).
Author: H. Mezni (original) / fix for Colab execution
"""

import random
import logging
import torch
from typing import List, Dict

logger = logging.getLogger(__name__)


class MetapathSampler:
    """
    Samples neighbors for each node following predefined trust meta-paths.
    N_v^(m) = {u in V | (u,v) in E and u->v follows m in M}
    """

    def __init__(self, tkg, metapaths: dict, sampling_size: int = 20, seed: int = 42):
        """
        tkg: TrustKnowledgeGraph instance
        metapaths: dict {name: {relations: [...]}}
        sampling_size: max neighbors per metapath per node
        """
        self.tkg = tkg
        self.metapaths = metapaths
        self.sampling_size = sampling_size
        random.seed(seed)

    def sample_node(self, node_id: str, mp_relations: List[str]) -> List[str]:
        """
        Follow mp_relations from node_id, return reachable end nodes (sampled).
        """
        current = [node_id]
        for rel in mp_relations:
            next_nodes = []
            for n in current:
                nbrs = self.tkg.get_neighbors_by_relation(n, rel)
                next_nodes.extend(nbrs)
            current = list(set(next_nodes))
            if not current:
                return []

        # Remove self
        if node_id in current:
            current.remove(node_id)

        # Random sampling
        if len(current) > self.sampling_size:
            current = random.sample(current, self.sampling_size)
        return current

    def build_neighborhoods(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Build neighborhoods for ALL nodes across ALL metapaths.
        Returns: {node_id: {mp_name: [neighbor_ids]}}
        """
        logger.info(f"Building metapath neighborhoods for {self.tkg.G.number_of_nodes()} nodes ...")
        neighborhoods = {}
        nodes = list(self.tkg.G.nodes())
        mp_names = list(self.metapaths.keys())

        for i, node_id in enumerate(nodes):
            neighborhoods[node_id] = {}
            for mp_name in mp_names:
                mp_rels = self.metapaths[mp_name]["relations"]
                nbrs = self.sample_node(node_id, mp_rels)
                neighborhoods[node_id][mp_name] = nbrs

            if (i + 1) % 500 == 0:
                logger.info(f"  Sampled {i+1}/{len(nodes)} nodes ...")

        logger.info("Neighborhood sampling complete.")
        return neighborhoods

    def to_index_neighborhoods(self, neighborhoods: Dict[str, Dict[str, List[str]]]) -> list:
        """
        Convert string neighborhoods to integer index neighborhoods.
        Returns list[node_idx][mp_idx] -> list of neighbor node indices.
        (Kept for backward compatibility / inspection; training itself now
        uses `to_padded_tensors` below for speed.)
        """
        node_index = self.tkg.node_index
        mp_names = list(self.metapaths.keys())
        nodes = [self.tkg.index_node[i] for i in range(len(self.tkg.node_index))]

        idx_neighborhoods = []
        for node_id in nodes:
            mp_list = []
            for mp_name in mp_names:
                nbr_ids = neighborhoods.get(node_id, {}).get(mp_name, [])
                nbr_idxs = [node_index[n] for n in nbr_ids if n in node_index]
                mp_list.append(nbr_idxs)
            idx_neighborhoods.append(mp_list)

        return idx_neighborhoods

    def to_padded_tensors(self, neighborhoods: Dict[str, Dict[str, List[str]]]):
        """
        Build padded (num_metapaths, N, K) index + mask tensors from the
        string-keyed neighborhoods, where K = self.sampling_size.

        Returns:
            nbr_idx_stack:  LongTensor (M, N, K) - neighbor node indices
                            (padded with 0; padding is masked out, never used
                            for real aggregation weight).
            nbr_mask_stack: BoolTensor (M, N, K) - True where the slot holds a
                            real neighbor.
        """
        node_index = self.tkg.node_index
        mp_names = list(self.metapaths.keys())
        N = len(node_index)
        K = self.sampling_size
        M = len(mp_names)
        nodes = [self.tkg.index_node[i] for i in range(N)]

        nbr_idx_stack = torch.zeros((M, N, K), dtype=torch.long)
        nbr_mask_stack = torch.zeros((M, N, K), dtype=torch.bool)

        for mp_i, mp_name in enumerate(mp_names):
            for v_idx, node_id in enumerate(nodes):
                nbr_ids = neighborhoods.get(node_id, {}).get(mp_name, [])
                nbr_idxs = [node_index[n] for n in nbr_ids if n in node_index][:K]
                if not nbr_idxs:
                    continue
                cnt = len(nbr_idxs)
                nbr_idx_stack[mp_i, v_idx, :cnt] = torch.tensor(nbr_idxs, dtype=torch.long)
                nbr_mask_stack[mp_i, v_idx, :cnt] = True

        return nbr_idx_stack, nbr_mask_stack
