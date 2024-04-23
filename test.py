import torch
import numpy as np
import argparse

import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

from model.gcn import GCN
from model.gat import GAT
from model.appnp import APPNP
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gat', 'appnp'], help='gnn model')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'pubmed', 'cora_full', 'reddit'], help='dataset')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load dataset
print(f'Loading {args.dataset} dataset')
dataset_file = f'data/{args.dataset}.npz'
data = load_and_standardize(dataset_file)

# Setup model
if args.model == 'gcn':
    model = GCN(nfeat=data.num_features, nhid=16, nclass=dataset.num_classes, device=device)
elif args.model == 'gat':
    model = GAT(nfeat=data.num_features, nhid=2, heads=8, nclass=dataset.num_classes, device=device)
elif args.model == 'appnp':
    model = APPNP(nfeat=data.num_features, nhid=16, K=8, alpha=0.15, nclass=dataset.num_classes, device=device)
model = model.to(device)

# Traing
model.fit(data=data) # train with earlystopping
model.test() # test

