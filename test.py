import torch
import numpy as np
import argparse
import warnings

import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

from model.gcn import GCN
from model.gat import GAT
from model.appnp import APPNP
from utils import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gat', 'appnp'], help='gnn model')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'pubmed', 'cora_full', 'cora_ml', 'reddit', 
                                                                    'amazon_computers', 'amazon_photo', 'ms_cs', 'ms_phy'], help='dataset')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load dataset
print(f'Loading {args.dataset} dataset')
dataset_file = f'data/{args.dataset}.npz'
graph = load_and_standardize(dataset_file, standard={'make_unweighted': True, 'make_undirected': True, 'no_self_loops': True, 'select_lcc': False})
print(f'Number of nodes: {graph.num_nodes()}')

data = SpG2PyG(graph, random_seed=args.seed)

# Setup model
if args.model == 'gcn':
    model = GCN(nfeat=graph.num_node_attr, nhid=16, nclass=graph.num_classes, device=device)
elif args.model == 'gat':
    model = GAT(nfeat=graph.num_node_attr, nhid=2, heads=8, nclass=graph.num_classes, device=device)
elif args.model == 'appnp':
    model = APPNP(nfeat=graph.num_node_attr, nhid=16, K=8, alpha=0.15, nclass=graph.num_classes, device=device)
model = model.to(device)

# Traing
model.fit(data=data) # train with earlystopping
model.test() # test

