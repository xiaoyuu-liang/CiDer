import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
import hydra
import pathlib
from omegaconf import DictConfig
import scipy.sparse as sp

from src.model.classifier import GCN
from src.general_utils import load_and_standardize, SpG2PyG, visualize_graph
from src.sparse_graph import create_subgraph

from src.model.diffusion.attributed_dataset import AttributedGraphDataModule, AttributedDatasetInfos

standard = {'make_unweighted': True,
            'make_undirected': True,
            'no_self_loops': True,
            'select_lcc': False}


@hydra.main(version_base='1.3', config_path='configs', config_name='config')
def main(cfg: DictConfig):
    dataset_name = 'pubmed'
    print(f'Dataset name: {dataset_name}')

    file = f'data/{dataset_name}.npz'
    data = np.load(file, allow_pickle=True)

    graph = load_and_standardize(file, standard)
    print(f'Number of nodes in graph: {graph.num_nodes()}')
    print(f'Number of edges in graph: {graph.num_edges()}')
    print(f'Number of classes: {graph.num_classes}')
    print(f'Number of node attributes: {graph.num_node_attr}')

    # np.set_printoptions(threshold=np.inf)
    # print(f'Number of nodes in graph: {graph.num_nodes()}')
    # # print(graph.attr_matrix[:30, :40])
    # num_zeros_per_attr = np.count_nonzero(graph.attr_matrix.toarray() == 0, axis=0)
    # print(f'Number of node attributes: {graph.num_node_attr, len(num_zeros_per_attr)}')
    # # print(f'Number of zeros per attribute: {num_zeros_per_attr}')
    # print(f'Number of classes: {graph.num_classes}')
    
    # degrees = graph.degrees
    # plt.hist(degrees, bins='auto')  # Choose an appropriate number of bins or 'auto'
    # plt.title(f"{dataset_name} Degree Distribution")
    # plt.xlabel("Degree")
    # plt.ylabel("Frequency")
    # plt.savefig('degree_distribution', dpi=300, bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':
    main()

