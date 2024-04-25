import torch
import random
import numpy as np
import argparse
import warnings

from sparse_graph import SparseGraph
from utils import load_and_standardize, SpG2PyG
from hierarchical_rand_pruning import hierarchical_rand_pruning

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'pubmed', 'cora_full', 'cora_ml', 'reddit', 
                                                                    'amazon_computers', 'amazon_photo', 'ms_cs', 'ms_phy'], help='dataset')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f'Loading {args.dataset} dataset')
dataset_file = f'data/{args.dataset}.npz'
graph = load_and_standardize(dataset_file, standard={'make_unweighted': True, 'make_undirected': True, 'no_self_loops': True, 'select_lcc': True})
print(f'Number of nodes: {graph.num_nodes()}')

global_random_state = np.random.RandomState(args.seed)

target_node = global_random_state.choice(graph.num_nodes(), 1)[0]

print(f'Target node: {target_node}, name: {graph.node_names[target_node]}')

layer_count = [1, 2]
injection_budget = (2, 0.1)

random_state = np.random.RandomState(args.seed+random.randint(1, 100))
subgraph = hierarchical_rand_pruning(graph=graph, target_node=target_node, layer_count=layer_count, 
                                     injection_budget=injection_budget, random_state=random_state)
print(f'Target node: {subgraph.target_node}, name: {subgraph.node_names[subgraph.target_node]}')
