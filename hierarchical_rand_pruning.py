import math
import torch
import numpy as np
from typing import Tuple

from sparse_graph import SparseGraph, create_subgraph
from utils import load_and_standardize, SpG2PyG


def hierarchical_rand_pruning(graph: SparseGraph, target_node: int, layer_count: list, 
                              injection_budget: Tuple[int, float],
                              random_state: np.random.RandomState):
    """
    Hierarchical random pruning.

    Parameters
    ----------
    graph : SparseGraph
        The graph.
    target_node : int
        The target node.
    layer_count : list
        The number of possible GNN layer counts.
    injection_budget: Tuple[int, float]
        delta_d: int 
            injected node degree budget
        delta_v: float
            injected node count budget (ratio)
    random_state : np.random.RandomState
        The random state.

    Returns
    -------
    subgraph : SparseGraph
        The pruned graph.
    target_node : int
        The target node in subgraph.
    """
    # randomly select the number of layers
    hop = random_state.choice(layer_count, 1)[0]
    print(f'Extracting {hop}-hop subgraph with HiRP.')

    # extract L-hop subgraph
    nodes_to_keep = set([target_node])
    for _ in range(hop):
        l_hop = set()
        for v in nodes_to_keep:
            neighbors = graph.get_neighbors(v)
            l_hop.update(neighbors)
        nodes_to_keep.update(l_hop)
    l_hop_graph = create_subgraph(graph, nodes_to_keep=nodes_to_keep, target_node=target_node)
    target_node = l_hop_graph.target_node
    print(f'Created {hop}-hop-subgraph with {l_hop_graph.num_nodes()} nodes and {l_hop_graph.num_edges()} edges with HiRP.')

    # randomly prune at most delta_v nodes with at most delta_d degree
    pruned_cnt = 0
    delta_d, delta_v = injection_budget
    delta_v = math.ceil(delta_v * l_hop_graph.num_nodes())

    nodes_to_remove = set()
    for d in range(delta_d):
        d_degree_indices = set(np.where(l_hop_graph.degrees == d+1)[0])
        d_degree_indices.discard(target_node)
        if pruned_cnt + len(d_degree_indices) < delta_v:
            # if the number of nodes with degree d is less than delta_v, prune all
            nodes_to_remove.update(d_degree_indices)
            pruned_cnt += len(d_degree_indices)
        else:
            # randomly prune delta_v - pruned_cnt nodes with degree d
            nodes_to_remove.update(random_state.choice(np.array(list(d_degree_indices)), delta_v - pruned_cnt, replace=False))
            pruned_cnt = delta_v

    subgraph = create_subgraph(l_hop_graph, nodes_to_remove=nodes_to_remove, target_node=target_node)

    print(f'Created subgraph with {subgraph.num_nodes()} nodes and {subgraph.num_edges()} edges with HiRP.')
    return subgraph
    