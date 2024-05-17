import torch
import numpy as np
import matplotlib.pyplot as plt
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
    dataset_config = cfg["dataset"]

    datamodule = AttributedGraphDataModule(cfg)
    dataset_infos = AttributedDatasetInfos(datamodule, dataset_config)

    fig, axs = plt.subplots(15, 1, figsize=(15, 30))  # Create a grid of 2 rows and 5 columns

    i = 0
    for _ in range(15):
        for data in datamodule.train_dataloader():
            print(data)
            break
        # label_node_attr = data.x.argmax(-1)
        # node_attr = np.array(label_node_attr)
        # axs[i].imshow(node_attr, cmap='gray_r', aspect='auto')
        # i += 1
        # if i == 15:
        #     break

    # plt.ylabel('Node Index')
    # plt.xlabel('Binary Node Attribute')
    # # Display the plot
    # plt.savefig('figs/cora_ego_attr_binary_heatmap.png', dpi=500, bbox_inches='tight')
    # plt.show()

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