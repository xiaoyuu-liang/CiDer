import seml.experiment
import torch
import random
import numpy as np
import argparse
import warnings
import logging

from sparse_graph import SparseGraph
from utils import load_and_standardize, SpG2PyG, get_marginal, get_one_hot
from hierarchical_rand_pruning import hierarchical_rand_pruning

from sacred import Experiment
import seml

ex = Experiment()
seml.setup_logger(ex)

ex = Experiment()
seml.setup_logger(ex)


@ex.config
def config():
    seed = 42
    dataset = 'cora'


@ex.automain
def run(seed, dataset, _run, _log):  

    warnings.filterwarnings("ignore")
    global_random_state = np.random.RandomState(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'Using device {device}')

    graph = load_and_standardize(dataset=dataset, 
                                 standard={'make_unweighted': True, 
                                           'make_undirected': True, 
                                           'no_self_loops': True, 
                                           'select_lcc': False})
    # global_random_state = np.random.RandomState(seed)
    # target_node = global_random_state.choice(graph.num_nodes(), 1)[0]
    # graph = hierarchical_rand_pruning(graph, target_node=target_node, layer_count=[1], injection_budget=(0, 0), random_state=global_random_state)

    attr_one_hot, adj_one_hot = get_one_hot(graph)
    np.set_printoptions(threshold=np.inf)
    attr_margin, label_margin, adj_margin = get_marginal(graph)
    
    # print(attr_one_hot.shape)
    # print('---------------------------------')
    # print(adj_one_hot.shape)

    print(attr_margin, label_margin, adj_margin)
    

