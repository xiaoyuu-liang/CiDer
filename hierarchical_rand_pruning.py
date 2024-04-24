import torch
import numpy as np
import argparse

from sparse_graph import SparseGraph, create_subgraph
from utils import load_and_standardize, SpG2PyG


def hierarchical_rand_pruning(graph: SparseGraph, target_node: int, layer_count: list, random_state: np.random.RandomState):
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
    random_state : np.random.RandomState
        The random state.

    Returns
    -------
    subgraph : SparseGraph
        The pruned graph.
    target_node : int
        The target node in subgraph.
    """
    # randomlly select the number of layers
    hop = random_state.choice(layer_count, 1)[0]
    # TODO: remember target node

    # extract L-hop
    nodes_to_keep = set([target_node])
    for l in range(hop):
        l_hop = set()
        for v in nodes_to_keep:
            neighbors = graph.get_neighbors(v)
            l_hop.update(neighbors)
        nodes_to_keep.update(l_hop)
    nodes_to_keep = np.array(list(nodes_to_keep))
    l_hop_graph = create_subgraph(graph, nodes_to_keep=nodes_to_keep)

    return target_node, l_hop_graph
    