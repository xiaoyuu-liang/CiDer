import os
import pathlib
import numpy as np

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import random_split
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.loader import DataLoader
from torch_geometric.data.lightning import LightningDataset

from .distribution import DistributionNodes
from .utils import to_dense, get_one_hot, get_marginal
from src.general_utils import load_and_standardize
from src.hierarchical_rand_pruning import hierarchical_rand_pruning


class AbstractDataModule(LightningDataset):
    def __init__(self, cfg, datasets):
        super().__init__(train_dataset=datasets['train'], val_dataset=datasets['val'], test_dataset=datasets['test'],
                         batch_size=cfg.train.batch_size if 'debug' not in cfg.general.name else 1,
                         num_workers=cfg.train.num_workers,
                         pin_memory=getattr(cfg.dataset, "pin_memory", False))
        self.cfg = cfg
        self.input_dims = None
        self.output_dims = None

    def __getitem__(self, idx):
        return self.train_dataset[idx]
    
    def graph_info(self):
        attr_margin, label_margin, adj_margin = get_marginal(self.train_dataset.graph)
        return attr_margin, label_margin, adj_margin
    
    def node_counts(self, max_nodes_possible=500):
        all_counts = torch.zeros(max_nodes_possible)
        for loader in [self.train_dataloader(), self.val_dataloader()]:
            for data in loader:
                unique, counts = torch.unique(data.batch, return_counts=True)
                for count in counts:
                    all_counts[count] += 1
        max_index = max(all_counts.nonzero())
        all_counts = all_counts[:max_index + 1]
        all_counts = all_counts / all_counts.sum()
        return all_counts


class AbstractDatasetInfos:
    def complete_infos(self, n_nodes, node_types, node_attr):
        self.input_dims = None
        self.output_dims = None
        self.num_classes = len(node_types)
        self.max_n_nodes = len(n_nodes) - 1
        self.num_node_attr = len(node_attr)
        self.nodes_dist = DistributionNodes(n_nodes)

    def compute_input_output_dims(self, datamodule):
        example_batch = next(iter(datamodule.train_dataloader()))
        ex_dense, node_mask = to_dense(example_batch.x, example_batch.edge_index, 
                                       example_batch.edge_attr, example_batch.batch)

        self.input_dims = {'X': example_batch['x'].size(1),
                           'Xc': example_batch['x'].size(2),
                           'E': example_batch['edge_attr'].size(1),
                           'y': example_batch['y'].size(1) + 1}      # + 1 due to time conditioning

        self.output_dims = {'X': example_batch['x'].size(1),
                            'Xc': example_batch['x'].size(2),
                            'E': example_batch['edge_attr'].size(1),
                            'y': 0}


class AttributedGraphDataset(InMemoryDataset):
    def __init__(self, dataset_name, split, root, transform=None, pre_transform=None, pre_filter=None,
                 standard = {'make_unweighted': True,
                             'make_undirected': True,
                             'no_self_loops': True,
                             'select_lcc': False}):
        base_dir = pathlib.Path(os.path.abspath(__file__)).parents[3]
        self.file = os.path.join(base_dir, f'data/{dataset_name}.npz')
        self.dataset_name = dataset_name
        self.split = split
        self.graph = load_and_standardize(self.file, standard=standard)
        self.num_graphs = self.graph.num_nodes()

        test_len = int(round(self.num_graphs * 0.2))
        train_len = int(round((self.num_graphs - test_len) * 0.8))
        val_len = self.num_graphs - train_len - test_len
        self.split_len = {'train': train_len, 'val': val_len, 'test': test_len}
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self):
            return [self.split + '.pt']

    def download(self):
        print('downloading and generating subgraphs')
        data_list = []
        for idx in range(self.graph.num_nodes()):
            for hop in [1]:
                egograph = hierarchical_rand_pruning(graph=self.graph, target_node=idx, layer_count=[hop],
                                                     injection_budget=(0, 0), random_state=np.random.RandomState(0))
                attr_one_hot, _ = get_one_hot(egograph)
                edge_index = torch.LongTensor(np.array(egograph.adj_matrix.nonzero()))
                edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
                edge_attr[:, 1] = 1
                y = torch.zeros([1, 0]).float()
                num_nodes = egograph.num_nodes() * torch.ones(1, dtype=torch.long)
                data = torch_geometric.data.Data(x=attr_one_hot, # (N, F, 2)
                                                 edge_index=edge_index, # 2 * |E| (sparse)
                                                 edge_attr=edge_attr, # ｜E｜ * 2
                                                 labels=egograph.labels,
                                                 y=y, 
                                                 n_nodes=num_nodes, 
                                                 target_node=egograph.target_node)
                
                data_list.append(data)
        print(f'Loaded {len(data_list)} graphs')
        
        g_cpu = torch.Generator()
        g_cpu.manual_seed(0)

        indices = torch.randperm(self.num_graphs, generator=g_cpu)
        train_indices = indices[:self.split.split_len['train']]
        val_indices = indices[self.split_len['train']:self.split.split_len['train'] + self.split.split_len['val']]
        test_indices = indices[self.split_len['train'] + self.split_len['val']:]

        train_data = []
        val_data = []
        test_data = []

        for i, data in enumerate(data_list):
            if i in train_indices:
                train_data.append(data)
            elif i in val_indices:
                val_data.append(data)
            elif i in test_indices:
                test_data.append(data)
            else:
                raise ValueError(f'Index {i} not in any split')
        
        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])
        torch.save(test_data, self.raw_paths[2])
        
        
    def process(self):
        file_idx = {'train': 0, 'val': 1, 'test': 2}
        raw_dataset = torch.load(self.raw_paths[file_idx[self.split]])

        data_list = []
        for data in raw_dataset:
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])



class AttributedGraphDataModule(AbstractDataModule):
    def __init__(self, cfg, n_graphs=5000):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[3]
        root_path = os.path.join(base_path, self.datadir)[1:]

        datasets = {'train': AttributedGraphDataset(dataset_name=self.cfg.dataset.name,
                                                    split='train', root=root_path),
                    'val': AttributedGraphDataset(dataset_name=self.cfg.dataset.name,
                                                  split='val', root=root_path),
                    'test': AttributedGraphDataset(dataset_name=self.cfg.dataset.name,
                                                   split='test', root=root_path)}
        # print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]


class AttributedDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'undirected_unweighted_binary_attributed_graph'
        self.node_counts = self.datamodule.node_counts() # node count marginal distribution
        attr_margin, label_margin, adj_margin = self.datamodule.graph_info()
        self.node_attrs = attr_margin # node attributes marginal distribution
        self.node_types = label_margin # node label marginal distribution
        self.edge_types = adj_margin # edge existence marginal distribution
        super().complete_infos(self.node_counts, self.node_types, self.node_attrs)


class HiRPDataset(AttributedGraphDataset):
    def __init__(self, dataset_name, split,root, transform=None, pre_transform=None, pre_filter=None,
                 hops=[1, 2], injection_budget=(0, 0.1), seed=42,
                 standard = {'make_unweighted': True,
                             'make_undirected': True,
                             'no_self_loops': True,
                             'select_lcc': False}):
        self.hops = hops
        self.injection_budget = injection_budget
        self.seed = seed
        super().__init__(dataset_name, split, root, transform, pre_transform, pre_filter, standard)
        self.split_len = {'train': 0, 'val': 0, 'test': self.num_graphs}
    
    @property
    def raw_file_names(self):
        return ['hirp_train.pt', 'hirp_val.pt', 'hirp_test.pt']

    @property
    def processed_file_names(self):
            return ['hirp_' + self.split + '.pt']
    
    def download(self):
        print('downloading and hierarchical random pruning subgraphs')
        data_list = []
        for idx in range(self.graph.num_nodes()):
            egograph = hierarchical_rand_pruning(graph=self.graph, target_node=idx, layer_count=self.hops,
                                                 injection_budget=(0, 0), random_state=np.random.RandomState(self.seed))
            attr_one_hot, _ = get_one_hot(egograph)
            edge_index = torch.LongTensor(np.array(egograph.adj_matrix.nonzero()))
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1
            y = torch.zeros([1, 0]).float()
            num_nodes = egograph.num_nodes() * torch.ones(1, dtype=torch.long)
            data = torch_geometric.data.Data(x=attr_one_hot, # (N, F, 2)
                                             edge_index=edge_index, # 2 * |E| (sparse)
                                             edge_attr=edge_attr, # ｜E｜ * 2
                                             labels=egograph.labels,
                                             y=y,
                                             n_nodes=num_nodes,
                                             target_node=egograph.target_node)
                
            data_list.append(data)
        print(f'Loaded {len(data_list)} graphs')
        
        torch.save(None, self.raw_paths[0])
        torch.save(None, self.raw_paths[1])
        torch.save(data_list, self.raw_paths[2]) # HiRP dataset is only for test
        
    def process(self):
        raw_dataset = torch.load(self.raw_paths[2])

        data_list = []
        for data in raw_dataset:
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])


class HiRPDataModule(AbstractDataModule):
    def __init__(self, cfg, n_graphs=5000, hops=[1, 2], injection_budget=(0, 0.1), seed=42):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[3]
        root_path = os.path.join(base_path, self.datadir)[1:]

        datasets = {'train': None,
                    'val': None,
                    'test': HiRPDataset(dataset_name=self.cfg.dataset.name, split='test', root=root_path,
                                        hops=hops, injection_budget=injection_budget, seed=seed)}
        # print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')    
        super().__init__(cfg, datasets)
        self.inner = self.test_dataset
    
    def test_dataloader(self):
        return DataLoader(self.inner, batch_size=1)

    def __getitem__(self, item):
        return self.inner[item]

