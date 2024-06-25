import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import argparse

# checkpoints/cora/async_pilot/test_only/gcn_cora_[1-X0.01-0.65-E0.00-0.00].pth
# checkpoints/cora/async_pilot/test_only/gcn_cora_[1-X0.00-0.00-E0.00-0.66].pth
# checkpoints/cora/async_pilot/test_only/gcn_cora_[1-X0.01-0.65-E0.00-0.66].pth

# rand_gnn_checkpoints/gcn_cora/gcn_cora_[1-X0.00-0.00-E0.00-0.66].pth
# rand_gnn_checkpoints/gcn_cora/gcn_cora_[1-X0.01-0.65-E0.00-0.00].pth
# rand_gnn_checkpoints/gcn_cora/gcn_cora_[1-X0.01-0.65-E0.00-0.66].pth

# rand_results/cora/GCN_cora_[0.8-0.01-0.65].pth


def parse_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument("--cert", type=str, required=True,help="path to the CiDer certificate file")
    arg.add_argument("--joint", type=str, required=False, default='', help="joint certificate slice")
    arg.add_argument("--singular", type=str, required=False, default='', help="singular certificate")
    
    args = vars(arg.parse_args())

    joint = args["joint"].split(',') if args["joint"] else []
    joint = (joint[0], int(joint[1]), int(joint[2])) if joint else ()

    return args["cert"], joint, args["singular"]

def main():
    path, joint, singular = parse_arguments()
    
    cert = torch.load(path)['multiclass']['cert_acc']

    if joint:
        max_ra_adj, max_rd_adj, max_ra_att, max_rd_att = cert[0]
        print(f'max radius for joint certificate: {max_ra_adj-2, max_rd_adj-2, max_ra_att-2, max_rd_att-2}')
        x_coords_adj, y_coords_adj, x_coords_att, y_coords_att = cert[1]
        cert_acc = cert[2]

        heatmap = np.zeros((max_ra_adj, max_rd_adj, max_ra_att, max_rd_att))
        for x_adj, y_adj, x_att, y_att, acc in zip(x_coords_adj, y_coords_adj, x_coords_att, y_coords_att, cert_acc):
            heatmap[x_adj, y_adj, x_att, y_att] = acc
        
        if joint[0] == 'adj':
            heatmap = heatmap[joint[1], joint[2], :, :]
        elif joint[0] == 'att':
            heatmap = heatmap[:, :, joint[1], joint[2]]
        else:
            raise ValueError("joint certificate slice must be either 'adj' or 'att'")
    else:
        max_ra, max_rd = cert[0]
        x_coords, y_coords = cert[1]
        acc = cert[2]

        heatmap = np.zeros((max_ra, max_rd))
        for x, y, acc in zip(x_coords, y_coords, acc):
            heatmap[x, y] = acc
    
    # scale_factor = 10**2
    # heatmap = heatmap * scale_factor

    # Define your custom colors
    colors = ["#5385bd", "#9bc7df", "#dff3f8"] 
    n_bins = 100  # Increase this number for a smoother transition between colors
    cmap_name = "SkyCmap"

    # Create the colormap
    sky_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    plt.figure(figsize=(6, 2))
    ax = sns.heatmap(heatmap, annot=True, cmap=sky_cmap, fmt=".2f", cbar_kws={'label': 'Accuracy gap'})
    if joint:
        if joint[0] == 'adj':
            ax.set_xlabel("Budget $\Delta_X^-$")
            ax.set_ylabel("Budget $\Delta_X^+$")
        elif joint[0] == 'att':
            ax.set_xlabel("Budget $\Delta_A^-$")
            ax.set_ylabel("Budget $\Delta_A^+$")
        else:
            raise ValueError("joint certificate slice must be either 'adj' or 'att'")
    elif singular == 'adj':
        ax.set_xlabel("Budget $\Delta_A^-$")
        ax.set_ylabel("Budget $\Delta_A^+$")
    elif singular == 'att':
        ax.set_xlabel("Budget $\Delta_X^-$")
        ax.set_ylabel("Budget $\Delta_X^+$")
    
    plt.gca().invert_yaxis()
    
    dir_name = os.path.dirname(path)
    parts = dir_name.split('/')
    ckpt = parts[-2]
    smoothing_config = os.path.splitext(os.path.basename(path))[0]

    if joint:
        save_name = f'figs/{ckpt}-{smoothing_config}-{joint[0]}-{joint[1]}-{joint[2]}.png'
    else:
        save_name = f'figs/{ckpt}-{smoothing_config}.png'
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()    

if __name__ == "__main__":
    
    main()

