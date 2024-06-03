import os
import torch
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F

from tqdm import tqdm
from torch_geometric.data import Data
from src.sparse_graph import SparseGraph
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer


def load_and_standardize(dataset_file, 
                         standard={'make_unweighted': True, 'make_undirected': True, 'no_self_loops':  True, 'select_lcc':  True}):
    """
    Run standardize() + make the attributes binary.

    Parameters
    ----------
    dataset_file
        Name of the file to load.
    standard
        default: {'make_unweighted': True, 'make_undirected': True, 'no_self_loops': True, 'select_lcc': True}
    Returns
    -------
    graph: SparseGraph
        The standardized graph

    """
    with np.load(dataset_file, allow_pickle=True) as loader:
        loader = dict(loader)
        if 'type' in loader:
            del loader['type']
        graph = SparseGraph.from_flat_dict(loader)
    
    graph.standardize(**standard)

    # binarize
    graph._flag_writeable(True)
    graph.adj_matrix[graph.adj_matrix != 0] = 1
    graph.attr_matrix[graph.attr_matrix != 0] = 1
    graph._flag_writeable(False)

    return graph


def binarize_labels(labels, sparse_output=False, return_classes=False):
    """
    Convert labels vector to a binary label matrix.

    In the default single-label case, labels look like
    labels = [y1, y2, y3, ...].
    Also supports the multi-label format.
    In this case, labels should look something like
    labels = [[y11, y12], [y21, y22, y23], [y31], ...].

    Parameters
    ----------
    labels : array-like, shape [num_samples]
        Array of node labels in categorical single- or multi-label format.
    sparse_output : bool, default False
        Whether return the label_matrix in CSR format.
    return_classes : bool, default False
        Whether return the classes corresponding to the columns of the label matrix.

    Returns
    -------
    label_matrix : np.ndarray or sp.csr_matrix, shape [num_samples, num_classes]
        Binary matrix of class labels.
        num_classes = number of unique values in "labels" array.
        label_matrix[i, k] = 1 <=> node i belongs to class k.
    classes : np.array, shape [num_classes], optional
        Classes that correspond to each column of the label_matrix.

    """
    if isinstance(labels[0], torch.Tensor) and labels[0].dim() > 0 and hasattr(labels[0], '__iter__'):  # labels[0] is iterable <=> multilabel format
        binarizer = MultiLabelBinarizer(sparse_output=sparse_output)
    else:
        binarizer = LabelBinarizer(sparse_output=sparse_output)
    label_matrix = binarizer.fit_transform(labels).astype(np.float32)
    return (label_matrix, binarizer.classes_) if return_classes else label_matrix


def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    num_samples, num_classes = labels.shape
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index, class_index] > 0.0:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])


def mask_to_index(index, size):
    all_idx = np.arange(size)
    return all_idx[index]


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask


def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples, num_classes = labels.shape
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate((train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices, :]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices, :]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices, :]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices


def SpG2PyG(graph: SparseGraph, random_seed, 
            split={'train_examples_per_class': 20, 
                   'val_examples_per_class': 30, 
                   'test_examples_per_class': None,
                   'train_size': None, 
                   'val_size': None, 
                   'test_size': None}):
    """
    Convert a SparseGraph object to PyG Data object.

    Parameters
    ----------
    graph: SparseGraph The graph to convert.
    random_seed: int Random seed for the split.
    split: dict The split to use. If None, use the default split.
        default: {'train_examples_per_class':20, 'val_examples_per_class': 30, 'test_examples_per_class': None}

    Returns
    -------
    PyG Data object
    """
    random_state = np.random.RandomState(random_seed)
    num_nodes = graph.num_nodes()

    edge_index = torch.LongTensor(graph.adj_matrix.nonzero())
    # by default, the features in pyg data is dense
    if sp.issparse(graph.attr_matrix):
        x = torch.FloatTensor(graph.attr_matrix.todense()).float()
    else:
        x = torch.FloatTensor(graph.attr_matrix).float()
    y = torch.LongTensor(graph.labels)

    bi_labels = binarize_labels(y)
    
    train_idx, val_idx, test_idx = get_train_val_test_split(random_state=random_state, labels=bi_labels, **split)

    data = Data(x=x, y=y, edge_index=edge_index, train_mask=index_to_mask(train_idx, num_nodes),
                val_mask=index_to_mask(val_idx, num_nodes), test_mask=index_to_mask(test_idx, num_nodes))

    return data


def visualize_graph(graph: SparseGraph, path='graphs/graph.png',
                    node_size = 15, font_size = 5, width = 0.5, figsize=(5, 5), dpi=300):
    """
    Visualize the graph.
    """

    import networkx as nx
    import matplotlib.pyplot as plt

    adj_matrix = graph.adj_matrix.tocoo()

    G = nx.from_scipy_sparse_matrix(adj_matrix)
    labels = {i: f'{graph.node_names[i]}' for i in range(graph.num_nodes())}

    # Draw the graph
    plt.figure(figsize=figsize)
    nx.draw(G, labels=labels, node_size=node_size, font_size=font_size, font_color='black', edge_color='gray', width=width)
    plt.savefig('graphs/subgraph.png', dpi=dpi)
    plt.show()


def classifier_predict(dataloader, classifier, device):
    """
    Predict with classifier on egograph dataset.
    """
    acc = 0
    correct = 0

    for data in tqdm(dataloader, desc="Predicting"):
        x = data.x.argmax(dim=-1).float().to(device)
        edge_index = data.edge_index.to(device)
        
        pred = classifier(x, edge_index)[data.target_node[0]]
        label = pred.argmax(-1)
        correct += (label == data.labels[0][data.target_node[0]])
            
    acc = correct / len(dataloader)
    print(f'Accuracy: {acc}')


def save_cetrificate(dict_to_save, dataset_config, hparams, path):

    arch = hparams['classifier']
    dataset = dataset_config['name']
    p = hparams['smoothing_config']['p']
    p_plus = hparams['smoothing_config']['p_plus']
    p_minus = hparams['smoothing_config']['p_minus']

    print(f'saving to {path}/{arch}_{dataset}_[{p}-{p_plus:.2f}-{p_minus:.2f}].pth')
    torch.save(dict_to_save, f'{path}/{arch}_{dataset}_[{p}-{p_plus:.2f}-{p_minus:.2f}].pth')
