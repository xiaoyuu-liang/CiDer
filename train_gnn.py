import seml.experiment
import torch
import numpy as np
import argparse
import warnings
import logging
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

from src.model.classifier import GCN, GAT, APPNP, SAGE
from src.general_utils import *


def train_gnn(model: str, dataset: str, seed: int, save_path: str, device: torch.device, 
              standard = {'make_unweighted': True, 'make_undirected': True, 'no_self_loops': True, 'select_lcc': False}):
    """
    Train GNN model on the dataset.

    Parameters
    ----------
    standard: dict
        The standardization to apply to load and standardize the graph.
    """
    # Load dataset
    print(f'Loading {dataset} dataset')
    graph = load_and_standardize(f'data/{dataset}.npz', standard=standard)
    print(f'Number of nodes: {graph.num_nodes()}')

    data = SpG2PyG(graph, random_seed=seed)

    # Setup model
    if model == 'gcn':
        model = GCN(nfeat=graph.num_node_attr, nlayers=1, nhid=16, nclass=graph.num_classes, device=device)
    elif model == 'gat':
        model = GAT(nfeat=graph.num_node_attr, nlayers=1, nhid=2, heads=8, nclass=graph.num_classes, device=device)
    elif model == 'appnp':
        model = APPNP(nfeat=graph.num_node_attr, nhid=16, K=8, alpha=0.15, nclass=graph.num_classes, device=device)
    elif model == 'sage':
        model = SAGE(nfeat=graph.num_node_attr, nhid=16, nclass=graph.num_classes, device=device)
    model.to(device)
    
    # Traing
    model.fit(data=data) # train with earlystopping
    model.test() # test

    # Save model
    torch.save(model.state_dict(), save_path)
    print(f'Model saved at {save_path}')


from sacred import Experiment
import seml

ex = Experiment()
seml.setup_logger(ex)

@ex.config
def config():
    seed = 0
    model = 'gcn'
    dataset = 'cora_ml'

@ex.automain
def run(seed, model, dataset, _run, _log):  

    warnings.filterwarnings("ignore")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'Using device {device}')
        
    train_gnn(model=model, dataset=dataset, seed=seed, 
            save_path=f'gnn_checkpoints/{model}_{dataset}_l1.pt', device=device)