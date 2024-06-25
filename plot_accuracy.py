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
    [0.7859, 0.74, 0.74, 0.74, 0.74], # adj 100
    [0.7860, 0.74, 0.74, 0.74, 0.74], # adj 200
    [0.7823, 0.74, 0.74, 0.8303, 0.74], # adj 300
    [0.7638, 0.74, 0.74, 0.74, 0.74], # adj 350
]

sparse_cora_acc = [
# attr  0,   100,  200,  300,  350
    [0.7356, 0.7387, 0.7469, 0.7233, 0.6594], # adj 0
    [0.7351, 0.74, 0.74, 0.74, 0.74], # adj 100
    [0.7342, 0.74, 0.74, 0.74, 0.74], # adj 200
    [0.6980, 0.74, 0.74, 0.7283, 0.74], # adj 300
    [0.6308, 0.74, 0.74, 0.74, 0.74], # adj 350
]

cider_coraml_acc = [
# attr  0,   100,  200,  300,  350 
    [0.74, 0.74, 0.74, 0.74, 0.74], # adj 0
    [0.74, 0.74, 0.74, 0.74, 0.74], # adj 100
    [0.74, 0.74, 0.74, 0.74, 0.74], # adj 200
    [0.74, 0.74, 0.74, 0.7913, 0.74], # adj 300
    [0.74, 0.74, 0.74, 0.74, 0.74], # adj 350
]

sparse_coraml_acc = [
# attr  0,   100,  200,  300,  350
    [0.74, 0.74, 0.74, 0.74, 0.74], # adj 0
    [0.74, 0.74, 0.74, 0.74, 0.74], # adj 100
    [0.74, 0.74, 0.74, 0.74, 0.74], # adj 200
    [0.74, 0.74, 0.74, 0.7866, 0.74], # adj 300
    [0.74, 0.74, 0.74, 0.74, 0.74], # adj 350
]

cider_citeseer_acc = [
# attr  0,   100,  200,  300,  350  
    [0.74, 0.74, 0.74, 0.74, 0.74], # adj 0
    [0.74, 0.74, 0.74, 0.74, 0.74], # adj 100
    [0.74, 0.74, 0.74, 0.74, 0.74], # adj 200
    [0.74, 0.74, 0.74, 0.6858, 0.74], # adj 300
    [0.74, 0.74, 0.74, 0.74, 0.74], # adj 350
]

sparse_citeseer_acc = [
# attr  0,   100,  200,  300,  350
    [0.74, 0.74, 0.74, 0.74, 0.74], # adj 0
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

