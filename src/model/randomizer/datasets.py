import torch
import numpy as np
from tqdm.auto import tqdm
import torch
import scipy.sparse as sp

from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from .sparse_smoothing.utils import split
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, random_split

from src.general_utils import load_and_standardize


def load_dataset(hparams, seed=42):
    name = hparams["dataset"]
    path = hparams["dataset_path"]

    if hparams["datatype"] == "graphs":
        return graph_inductive_split(*load_graph(name, path), seed=seed)


def load_graph(name, path):
    """Load graph data.
       Preprocessings: to undirected, extract largest connected component,
       and remove self-loops.

    Args:
        name (string): Dataset name. Supported: ["Cora", "Pubmed", "Citeseer"].
        path (string): Folder with graph datasets.
    """
    if name in ["cora_ml", "cora", "citeseer", "pubmed"]:
        graph = load_and_standardize(f"{path}{name}.npz")
        
        A = graph.adj_matrix
        X = np.matrix(graph.attr_matrix.toarray())
        y = graph.labels
        print(f'num nodes: {len(y)}, num attributes: {X.shape[1]}, num classes: {len(np.unique(y))}')

    return A, X, y


def graph_inductive_split(A, X, y, n_per_class=20, seed=42):
    n, d = X.shape
    nc = y.max() + 1

    # semi-supervised inductive setting
    idx_train, idx_valid, idx_unlabelled = split(
        labels=y, n_per_class=n_per_class, seed=seed)
    idx_unlabelled, idx_test = train_test_split(
        idx_unlabelled, test_size=0.1, random_state=seed)

    # graph splitting
    idx = np.hstack([idx_unlabelled, idx_train])
    train = (A[np.ix_(idx, idx)], X[idx, :], y[idx_train])

    idx = np.hstack([idx_unlabelled, idx_train, idx_valid])
    valid = (A[np.ix_(idx, idx)], X[idx, :], y[idx_valid])

    idx = np.hstack([idx_unlabelled, idx_train, idx_valid, idx_test])
    test = (A[np.ix_(idx, idx)], X[idx, :], y[idx_test])

    len1 = len(idx_unlabelled)
    len2 = len1+len(idx_train)
    len3 = len2+len(idx_valid)
    len4 = len3+len(idx_test)

    final_idx_train = np.arange(len1, len2)
    final_idx_valid = np.arange(len2, len3)
    final_idx_test = np.arange(len3, len4)

    return [A, X, y, n, d, nc, train, valid, test,
            final_idx_train, final_idx_valid, final_idx_test]


def prepare_graph_data(split, device='cuda'):
    edge_idx = torch.tensor(np.array(split[0].nonzero()), dtype=torch.long)
    x = torch.tensor(split[1])
    y = torch.LongTensor(split[2])
    return Data(A=split[0], x=x, edge_index=edge_idx, y=y).to(device)
