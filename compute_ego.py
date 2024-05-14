import torch
import numpy as np
import pickle
import hydra
from omegaconf import DictConfig

from src.model.classifier import GCN
from src.general_utils import load_and_standardize, SpG2PyG
from src.hierarchical_rand_pruning import hierarchical_rand_pruning

from src.model.diffusion.attributed_dataset import HiRPDataset, HiRPDataModule
from src.model.diffusion.attributed_dataset import AttributedGraphDataModule, AttributedDatasetInfos

standard = {'make_unweighted': True,
            'make_undirected': True,
            'no_self_loops': True,
            'select_lcc': False}


@hydra.main(version_base='1.3', config_path='configs', config_name='config')
def main(cfg: DictConfig):
    datamodule = HiRPDataModule(cfg)
    
    print(datamodule.test_dataloader().dataset.split_len['test'])

if __name__ == '__main__':
    main()


# classifier_path = 'gnn_checkpoints/gcn_cora.pt'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# file = 'data/cora.npz'
# graph = load_and_standardize(file, standard=standard)
# data = SpG2PyG(graph, random_seed=42)

# classifier = GCN(nfeat=data.num_features, nhid=16, nclass=graph.num_classes, device=device)
# classifier.load_state_dict(torch.load(classifier_path))

# classifier.eval()
# classifier.to(device)

# classifier.data = data.to(device)
# acc_test = classifier.test()
# print(f'test accuracy: {acc_test}')

# data_list = []
# for idx in range(graph.num_nodes()):
#     for hop in [1, 2]:
#         egograph = hierarchical_rand_pruning(graph=graph, target_node=idx, layer_count=[hop],
#                                              injectiacc_teston_budget=(0, 0), random_state=np.random.RandomState(0))
#         data_list.append(egograph)

# with open('ego2_cora.pkl', 'wb') as f:
#     pickle.dump(data_list, f)

# node_nums = [ego.num_nodes() for ego in data_list]
# print(node_nums)
# print(f'node nums mean: {np.mean(node_nums)}, max: {np.max(node_nums)}, min: {np.min(node_nums)}')