import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import argparse

datasets = ['cora', 'cora_ml', 'citeseer', 'pubmed']
naive_acc = [0.74, 0.71, 0.61, 0.72]

flip_prob_X = [0.00, 0.10, 0.35, 0.65, 0.79]
flip_prob_E = [0.00, 0.10, 0.35, 0.66, 0.80]
flip_prob_X_pubmed = [0.00, 0.01, 0.02, 0.59, 0.04]

cider_cora_acc = [
# attr  0,   100,  200,  300,  350  
    [0.7878, 0.7896, 0.7897, 0.8007, 0.7306], # adj 0
    [0.7859, 0.7878, 0.7970, 0.8044, 0.7306], # adj 100
    [0.7860, 0.8007, 0.8026, 0.8136, 0.7103], # adj 200
    [0.7823, 0.8007, 0.8100, 0.8210, 0.7140], # adj 300
    [0.7638, 0.7768, 0.8081, 0.8136, 0.7140], # adj 350
]

sparse_cora_acc = [
# attr  0,   100,  200,  300,  350
    [0.7356, 0.7387, 0.7469, 0.7233, 0.6594], # adj 0
    [0.7351, 0.7374, 0.7442, 0.7456, 0.6952], # adj 100
    [0.7342, 0.7342, 0.7401, 0.7347, 0.6268], # adj 200
    [0.6980, 0.7129, 0.7197, 0.7283, 0.7052], # adj 300
    [0.6308, 0.6630, 0.6957, 0.6975, 0.6893], # adj 350
]

cider_coraml_acc = [
# attr  0,   100,  200,  300,  350 
    [0.7663, 0.7696, 0.7763, 0.7730, 0.6945], # adj 0
    [0.7646, 0.7696, 0.7730, 0.7746, 0.6945], # adj 100
    [0.7663, 0.7679, 0.7730, 0.7813, 0.7112], # adj 200
    [0.7529, 0.7629, 0.7813, 0.7913, 0.6928], # adj 300
    [0.7412, 0.7529, 0.7713, 0.7830, 0.7045], # adj 350
]

sparse_coraml_acc = [
# attr  0,   100,  200,  300,  350
    [0.7933, 0.7767, 0.7767, 0.7885, 0.7703], # adj 0
    [0.7779, 0.7770, 0.7916, 0.7933, 0.7675], # adj 100
    [0.7822, 0.7826, 0.7940, 0.7909, 0.7866], # adj 200
    [0.7561, 0.7565, 0.7905, 0.7866, 0.7858], # adj 300
    [0.7090, 0.7368, 0.7660, 0.7628, 0.74], # adj 350
]

cider_citeseer_acc = [
# attr  0,   100,  200,  300,  350  
    [0.6299, 0.74, 0.74, 0.74, 0.74], # adj 0
    [0.74, 0.74, 0.74, 0.74, 0.74], # adj 100
    [0.74, 0.74, 0.74, 0.74, 0.74], # adj 200
    [0.74, 0.74, 0.74, 0.6858, 0.74], # adj 300
    [0.74, 0.74, 0.74, 0.74, 0.74], # adj 350
]

sparse_citeseer_acc = [
# attr  0,   100,  200,  300,  350
    [0.6588, 0.74, 0.74, 0.74, 0.74], # adj 0
    [0.74, 0.74, 0.74, 0.74, 0.74], # adj 100
    [0.74, 0.74, 0.74, 0.74, 0.74], # adj 200
    [0.74, 0.74, 0.74, 0.6652, 0.74], # adj 300
    [0.74, 0.74, 0.74, 0.74, 0.74], # adj 350
]

cider_pubmed_acc = [
# attr  0,   100,  200,  300,  350 
    [0.74, 0.74, 0.74, 0.74, 0.74], # adj 0
    [0.74, 0.74, 0.74, 0.74, 0.74], # adj 100
    [0.74, 0.74, 0.74, 0.74, 0.74], # adj 200
    [0.74, 0.74, 0.74, 0.7385, 0.74], # adj 300
    [0.74, 0.74, 0.74, 0.74, 0.74], # adj 350
]

sparse_pubmed_acc = [
# attr  0,   100,  200,  300, 350
    [0.74, 0.74, 0.74, 0.74, 0.74], # adj 0
    [0.74, 0.74, 0.74, 0.74, 0.74], # adj 100
    [0.74, 0.74, 0.74, 0.74, 0.74], # adj 200
    [0.74, 0.74, 0.74, 0.6901, 0.74], # adj 300
    [0.74, 0.74, 0.74, 0.74, 0.74], # adj 350
]

def parse_arguments():
    arg = argparse.ArgumentParser()

    args = vars(arg.parse_args())

    return

def main():
        
    acc_gap_sparse = np.array(cider_cora_acc) - np.array(sparse_cora_acc)
    acc_gap_naive = np.array(cider_cora_acc) - naive_acc[0]

    # Define your custom colors
    colors = ["#f7fef0", "#ceefcc", "#6fc8ca", "#3492b2"] 
    n_bins = 100  # Increase this number for a smoother transition between colors
    cmap_name = "MintCmap"

    # Create the colormap
    mint_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    # Draw heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(acc_gap_naive, annot=True, cmap=mint_cmap, fmt=".2f", xticklabels=flip_prob_X, yticklabels=flip_prob_E, annot_kws={"size":14})
    plt.xlabel("Flip Probability X", fontsize=20)
    plt.ylabel("Flip Probability E", fontsize=20)
    plt.gca().invert_yaxis()
    # Set color bar label font size
    cbar = ax.collections[0].colorbar
    cbar.set_label('Clean accuracy gap', size=14)

    # Set color bar tick label font size
    cbar.ax.tick_params(labelsize=10)

    plt.savefig("figs/naive_cora.png")
    plt.show()
    
    return
    

if __name__ == "__main__":
    
    main()

