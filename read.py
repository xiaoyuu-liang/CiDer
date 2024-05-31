import torch

dict_loaded = torch.load('rand_results/GCN_cora_[0.8-0.0-0.35].pth')

print(dict_loaded['majority_acc'])
