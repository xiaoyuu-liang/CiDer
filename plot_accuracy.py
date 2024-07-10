import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import argparse

datasets = ['cora', 'cora_ml', 'citeseer', 'pubmed']
seed = [42, 0, 78, 71]
naive_acc = [0.7375, 0.7928, 0.6829, 0.7694]

flip_prob_X_cora = [0.00, 0.10, 0.35, 0.65, 0.79]
flip_prob_X_citeseer = [0.00, 0.10, 0.35, 0.65, 0.79]
flip_prob_X_coraml = [0.00, 0.10, 0.35, 0.65, 0.78]
flip_prob_X_pubmed = [0.00, 0.09, 0.32, 0.59, 0.72]

flip_prob_E = [0.00, 0.10, 0.35, 0.66, 0.80]


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
    [0.7475, 0.7447, 0.7461, 0.7321, 0.7128], # adj 0
    [0.7400, 0.7461, 0.7513, 0.7480, 0.7087], # adj 100
    [0.7400, 0.7447, 0.7564, 0.7428, 0.7077], # adj 200
    [0.7293, 0.7368, 0.7415, 0.7466, 0.6295], # adj 300
    [0.6979, 0.7053, 0.7227, 0.6656, 0.5672], # adj 350
]

cider_coraml_acc = [
# attr  0,   100,  200,  300,  350 
    [0.8247, 0.8297, 0.8381, 0.8331, 0.7846], # adj 0
    [0.8214, 0.8247, 0.8331, 0.8347, 0.7963], # adj 100
    [0.8197, 0.8230, 0.8297, 0.8447, 0.7947], # adj 200
    [0.8280, 0.8247, 0.8297, 0.8397, 0.7997], # adj 300
    [0.8247, 0.8314, 0.8347, 0.8297, 0.8030], # adj 350
]

sparse_coraml_acc = [
# attr  0,   100,  200,  300,  350
    [0.7849, 0.7785, 0.7850, 0.7878, 0.7658], # adj 0
    [0.7695, 0.7732, 0.7821, 0.7910, 0.7699], # adj 100
    [0.7780, 0.7829, 0.7902, 0.7935, 0.7792], # adj 200
    [0.7683, 0.7873, 0.7955, 0.7923, 0.7545], # adj 300
    [0.7557, 0.7707, 0.7695, 0.7695, 0.4613], # adj 350
]

cider_citeseer_acc = [
# attr  0,   100,  200,  300,  350  
    [0.7221, 0.7266, 0.7311, 0.7221, 0.6178], # adj 0
    [0.7190, 0.7266, 0.7371, 0.7236, 0.6088], # adj 100
    [0.7296, 0.7372, 0.7387, 0.7251, 0.6057], # adj 200
    [0.7266, 0.7356, 0.7311, 0.7221, 0.5951], # adj 300
    [0.7160, 0.7341, 0.7251, 0.7130, 0.6103], # adj 350
]

sparse_citeseer_acc = [
# attr  0,   100,  200,  300,  350
    [0.6585, 0.6436, 0.6608, 0.6724, 0.6807], # adj 0
    [0.6409, 0.6448, 0.6547, 0.6696, 0.6845], # adj 100
    [0.6386, 0.6387, 0.6541, 0.6746, 0.6818], # adj 200
    [0.6199, 0.6403, 0.6586, 0.6745, 0.6856], # adj 300
    [0.5994, 0.6259, 0.6337, 0.6574, 0.6745], # adj 350
]

cider_pubmed_acc = [
# attr  0,   100,  200,  300,  350 
    [0.7652, 0.7704, 0.7763, 0.7601, 0.6820], # adj 0
    [0.7654, 0.7720, 0.7786, 0.7619, 0.6817], # adj 100
    [0.7664, 0.7735, 0.7829, 0.7682, 0.6837], # adj 200
    [0.7631, 0.7824, 0.7915, 0.7730, 0.6867], # adj 300
    [0.7492, 0.7781, 0.7903, 0.7723, 0.6817], # adj 350
]

sparse_pubmed_acc = [
# attr  0,   100,  200,  300, 350
    [0.7396, 0.7355, 0.7360, 0.7036, 0.6160], # adj 0
    [0.7358, 0.7297, 0.7331, 0.7168, 0.6523], # adj 100
    [0.7350, 0.7395, 0.7369, 0.7059, 0.4880], # adj 200
    [0.7122, 0.7213, 0.7238, 0.6271, 0.3680], # adj 300
    [0.7128, 0.6978, 0.6620, 0.6451, 0.3724], # adj 350
]

def parse_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument("--data", type=str, required=True,help="dataset name")
    arg.add_argument("--gap", type=str, required=True, default='', help="gap type")

    args = vars(arg.parse_args())

    return args["data"], args["gap"]

def main():

    dataset, gap = parse_arguments()
    
    if dataset == 'citeseer':
        acc_gap_sparse = np.array(cider_citeseer_acc) - np.array(sparse_citeseer_acc)
        acc_gap_naive = np.array(cider_citeseer_acc) - naive_acc[2]
        flip_prob_X = flip_prob_X_citeseer
    if dataset == 'cora':
        acc_gap_sparse = np.array(cider_cora_acc) - np.array(sparse_cora_acc)
        acc_gap_naive = np.array(cider_cora_acc) - naive_acc[0]
        flip_prob_X = flip_prob_X_cora
    if dataset == 'coraml':
        acc_gap_sparse = np.array(cider_coraml_acc) - np.array(sparse_coraml_acc)
        acc_gap_naive = np.array(cider_coraml_acc) - naive_acc[1]
        flip_prob_X = flip_prob_X_coraml
    if dataset == 'pubmed':
        acc_gap_sparse = np.array(cider_pubmed_acc) - np.array(sparse_pubmed_acc)
        acc_gap_naive = np.array(cider_pubmed_acc) - naive_acc[3]
        flip_prob_X = flip_prob_X_pubmed
    
    if gap == 'naive':
        acc_gap = acc_gap_naive
    if gap == 'sparse':
        acc_gap = acc_gap_sparse
    print(f'Average accuracy gap with {gap} for {dataset}: {np.mean(acc_gap):.6f}')

    # Define your custom colors
    colors = ["#fff4d4", "#f7fef0", "#ceefcc", "#6fc8ca", "#3492b2"] 
    n_bins = 100  # Increase this number for a smoother transition between colors
    cmap_name = "MintCmap"

    # Create the colormap
    mint_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    # Draw heatmap
    plt.figure(figsize=(10, 8.5))
    ax = sns.heatmap(acc_gap, annot=True, cmap=mint_cmap, fmt=".2f", 
                     xticklabels=flip_prob_X, yticklabels=flip_prob_E, annot_kws={"size":14},
                     vmin=-0.05, vmax=0.15,
                     cbar_kws={'ticks': np.linspace(-0.05, 0.15, 5)})
    plt.xlabel("X Flip Probability", fontsize=26, labelpad=20)
    plt.ylabel("A Flip Probability", fontsize=26, labelpad=20)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.tight_layout()
    plt.gca().invert_yaxis()
    # Set color bar label font size
    cbar = ax.collections[0].colorbar
    cbar.set_label('Clean accuracy gap', size=26)
    cbar.ax.yaxis.set_tick_params(labelsize=22)
    cbar.ax.tick_params(labelsize=22)

    plt.savefig(f"figs/{gap}_{dataset}.png")
    print(f"figs/{gap}_{dataset}.png saved")
    plt.show()
    
    return
    

if __name__ == "__main__":
    
    main()

