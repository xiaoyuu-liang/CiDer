import torch
import numpy as np
import pickle

from src.utils import load_and_standardize
from src.hierarchical_rand_pruning import hierarchical_rand_pruning

standard = {'make_unweighted': True,
            'make_undirected': True,
            'no_self_loops': True,
            'select_lcc': False}

file = 'data/cora.npz'
graph = load_and_standardize(file, standard=standard)

data_list = []
for idx in range(graph.num_nodes()):
    for hop in [1, 2]:
        egograph = hierarchical_rand_pruning(graph=graph, target_node=idx, layer_count=[hop],
                                             injection_budget=(0, 0), random_state=np.random.RandomState(0))
        data_list.append(egograph)

with open('ego2_cora.pkl', 'wb') as f:
    pickle.dump(data_list, f)

node_nums = [ego.num_nodes() for ego in data_list]
print(node_nums)
print(f'node nums mean: {np.mean(node_nums)}, max: {np.max(node_nums)}, min: {np.min(node_nums)}')