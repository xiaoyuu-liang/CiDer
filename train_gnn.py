import torch
import numpy as np
import argparse
import warnings

import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

from model.gcn import GCN
from model.gat import GAT
from model.appnp import APPNP
from model.sage import SAGE
from utils import *


def train_gnn(model: str, dataset: str, seed: int, save_path: str,
              device: torch.device, standard={}):
    """
    Train GNN model on the dataset.

    Parameters
    ----------
    standard: dict
        The standardization to apply to load and standardize the graph.
    """
    # Load dataset
    print(f'Loading {dataset} dataset')
    dataset_file = f'data/{dataset}.npz'
    graph = load_and_standardize(dataset_file, standard={'make_unweighted': True, 'make_undirected': True, 'no_self_loops': True, 'select_lcc': False})
    print(f'Number of nodes: {graph.num_nodes()}')

    data = SpG2PyG(graph, random_seed=seed)

    # Setup model
    if model == 'gcn':
        model = GCN(nfeat=graph.num_node_attr, nhid=16, nclass=graph.num_classes, device=device)
    elif model == 'gat':
        model = GAT(nfeat=graph.num_node_attr, nhid=2, heads=8, nclass=graph.num_classes, device=device)
    elif model == 'appnp':
        model = APPNP(nfeat=graph.num_node_attr, nhid=16, K=8, alpha=0.15, nclass=graph.num_classes, device=device)
    elif model == 'sage':
        model = SAGE(nfeat=graph.num_node_attr, nhid=16, nclass=graph.num_classes, device=device)
    model = model.to(device)

    # Traing
    model.fit(data=data) # train with earlystopping
    model.test() # test

    # Save model
    torch.save(model.state_dict(), save_path)


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gat', 'appnp', 'sage'], help='gnn model')
parser.add_argument('--dataset', type=str, default='pubmed', choices=['cora', 'citeseer', 'pubmed', 'cora_full', 'cora_ml', 'reddit', 
                                                                    'amazon_computers', 'amazon_photo', 'ms_cs', 'ms_phy'], help='dataset')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_gnn(model=args.model, dataset=args.dataset, seed=args.seed, save_path=f'checkpoints/{args.model}_{args.dataset}.pt', device=device)