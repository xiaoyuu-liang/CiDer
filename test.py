import torch
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
graph = load_and_standardize(dataset_file, standard={'make_unweighted': True, 'make_undirected': True, 'no_self_loops': True, 'select_lcc': False})
print(f'Number of nodes: {graph.num_nodes()}')

data = SpG2PyG(graph, random_seed=args.seed)

random_state = np.random.RandomState(args.seed)

target_node = random_state.choice(graph.num_nodes(), 1)[0]
print(f'Target node: {target_node}')

layer_count = [1, 2]

subgraph = hierarchical_rand_pruning(graph, target_node, layer_count, random_state)

import networkx as nx
import matplotlib.pyplot as plt

# Assuming `sparse_graph` is your SparseGraph instance
adj_matrix = subgraph.adj_matrix.tocoo()

# Create a new NetworkX graph from the adjacency matrix
G = nx.from_scipy_sparse_matrix(adj_matrix)
labels = {i: f'{subgraph.node_names[i]}' for i in range(subgraph.num_nodes())}
print(labels)

# Draw the graph
nx.draw(G, labels=labels)
plt.savefig('graphs/l-hop-subgraph.png')
plt.show()