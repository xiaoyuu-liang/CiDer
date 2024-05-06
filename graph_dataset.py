import torch
import logging
import numpy as np
from torch_geometric.data import Data, Dataset

from src.sparse_graph import SparseGraph
from utils import load_and_standardize
from src.model.hierarchical_rand_pruning import hierarchical_rand_pruning, SpG2PyG


class GraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 name='cora', random_seed=42,
                 standard={'make_unweighted': True, 'make_undirected': True, 'no_self_loops': True, 'select_lcc': False}):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.data_file = 'data/dataset/' + name + '.npz'
        self.standard = standard
        self.random_state = np.random.RandomState(random_seed)

        print(f'Loading {name} dataset')
        self.graph = load_and_standardize(self.data_file, standard=self.standard)
        logging.info(f'Number of nodes: {self.graph.num_nodes()}')
        

    @property
    def get_graph(self):
        return self.graph
    
    def extract_egograph(self, node_idx, hop=1):
        """
        Extract the egograph of hop of the node.

        Parameters
        ----------
        node_idx : int
            The node index.
        hop : int 1/2 (default=1)
        """
        # Extract the egograph of the node
        egograph = hierarchical_rand_pruning(graph=self.graph, target_node=node_idx, layer_count=[hop], 
                                             injection_budget=(0, 0), random_state=np.random.RandomState(0))
        
        return egograph
    
    def extract_ego_dataset(self, node_indices, hop=[1, 2]):
        """
        Extract the egograph dataset (1/2-hop egograph of all nodes).

        Parameters
        ----------
        node_indices : list
            The list of node indices.
        hop : list (default=[1, 2])
            The hop of the egograph.
        """
        for node_idx in node_indices:
            for h in hop:
                egograph = self.extract_egograph(node_idx, h)
                ego_lable = self.graph.node_label[node_idx]
                yield (egograph, ego_lable)
        
        return (egograph, ego_lable)
    
    