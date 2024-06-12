import torch
import os
import numpy as np
import matplotlib.pyplot as plt
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
    arg.add_argument("--cert_path", type=str, required=True,help="path to the certificate file")
    arg.add_argument("--op", type=str, required=True, help="operation to perform")
    arg.add_argument("--joint", type=str, required=False, default='', help="joint certificate slice")
    arg.add_argument("--h_radius", type=int, required=False, default=0, help="hierarchical certificate radius")
    
    args = vars(arg.parse_args())

    joint = args["joint"].split(',') if args["joint"] else []
    joint = (joint[0], int(joint[1]), int(joint[2])) if joint else ()

    return args["cert_path"], args["op"], args["h_radius"], joint

def main():
    cert_path, op, h_radius, joint= parse_arguments()
    
    certificate = torch.load(cert_path)
    data = certificate['multiclass'][op]

    if h_radius:
        data = data[0]       # value (data[1]=std)

        max_r, max_ra, max_rd = data[0]
        print(f'max radius for hierarchical certificate: {max_r-1}')
        radius, x_coords, y_coords = data[1]
        values = data[2]
        
        heatmap = np.zeros((max_r, max_ra, max_rd))
        for r, x, y, value in zip(radius, x_coords, y_coords, values):
            heatmap[r, x, y] = value
        heatmap = heatmap[h_radius]
    if joint:
        max_ra_adj, max_rd_adj, max_ra_att, max_rd_att = data[0]
        print(f'max radius for joint certificate: {max_ra_adj-2, max_rd_adj-2, max_ra_att-2, max_rd_att-2}')
        x_coords_adj, y_coords_adj, x_coords_att, y_coords_att = data[1]
        values = data[2]

        heatmap = np.zeros((max_ra_adj, max_rd_adj, max_ra_att, max_rd_att))
        for x_adj, y_adj, x_att, y_att, value in zip(x_coords_adj, y_coords_adj, x_coords_att, y_coords_att, values):
            heatmap[x_adj, y_adj, x_att, y_att] = value
        if op == 'ratios':
            heatmap[0, 0, 0, 0] = 1.0
        else:
            heatmap[0, 0, 0, 0] = certificate['clean_acc']
        
        if joint[0] == 'adj':
            heatmap = heatmap[joint[1], joint[2], :, :]
        elif joint[0] == 'att':
            heatmap = heatmap[:, :, joint[1], joint[2]]
        else:
            raise ValueError("joint certificate slice must be either 'adj' or 'att'")

    else:
        max_ra, max_rd = data[0]
        x_coords, y_coords = data[1]
        values = data[2]

        heatmap = np.zeros((max_ra, max_rd))
        for x, y, value in zip(x_coords, y_coords, values):
            heatmap[x, y] = value
        if op == 'ratios':
            heatmap[0, 0] = 1.0
        else:
            heatmap[0, 0] = certificate['clean_acc']

    plt.figure(figsize=(6, 2))
    ax = sns.heatmap(heatmap, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Value'})
    ax.set_title(f"Certificate {op} heatmap")
    ax.set_xlabel("Radius $r_d$")
    ax.set_ylabel("Radius $r_a$")
    plt.gca().invert_yaxis()
    
    dir_name = os.path.dirname(cert_path)
    parts = dir_name.split('/')
    ckpt = parts[-2]
    smoothing_config = os.path.splitext(os.path.basename(cert_path))[0]

    if h_radius:
        save_name = f'figs/{ckpt}-{smoothing_config}-{op}-h{h_radius}.png'
    if joint:
        save_name = f'figs/{ckpt}-{smoothing_config}-{op}-{joint[0]}-{joint[1]}-{joint[2]}.png'
    else:
        save_name = f'figs/{ckpt}-{smoothing_config}-{op}.png'
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()    

if __name__ == "__main__":
    
    main()

