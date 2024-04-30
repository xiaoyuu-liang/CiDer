import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import wandb
import warnings
import logging

from copy import deepcopy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.diffusion.diffuser import GraphJointDiffuser
from utils import load_and_standardize, get_marginal, get_one_hot
from graphmaker_diff.setup_utils import load_train_yaml


def train_diffuser(dataset):
    model_name = "GraphJointDiffuser"
    yaml_data = load_train_yaml(dataset)

    config_df = pd.json_normalize(yaml_data, sep='/')
    # Number of time steps
    T = yaml_data['diffusion']['T']
    
    wandb.init(
        project=f"{model_name}_{dataset}",
        name=f"T{T}",
        config=config_df.to_dict(orient='records')[0])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f'Loading {dataset} dataset')
    dataset_file = f'data/{dataset}.npz'
    graph = load_and_standardize(dataset_file, standard={}) # SparseGraph
    print(f'Number of nodes: {graph.num_nodes()}')

    attr_margin, label_margin, adj_margin = get_marginal(graph)
    attr_margin = attr_margin.to(device)
    label_margin = label_margin.to(device)
    adj_margin = adj_margin.to(device)

    X_one_hot_3d = X_one_hot_3d.to(device)
    Y = Y.to(device)
    E_one_hot = E_one_hot.to(device)

    train_config = yaml_data["train"]
    # For mini-batch training
    data_loader = DataLoader(edge_index.cpu(), batch_size=train_config["batch_size"],
                             shuffle=True, num_workers=4)
    val_data_loader = DataLoader(edge_index, batch_size=train_config["val_batch_size"],
                                 shuffle=False)

    model = GraphJointDiffuser(X_marginal=X_marginal,
                      Y_marginal=Y_marginal,
                      E_marginal=E_marginal,
                      num_nodes=N,
                      gnn_X_config=yaml_data["gnn_X"],
                      gnn_E_config=yaml_data["gnn_E"],
                      **yaml_data["diffusion"]).to(device)

    optimizer_X = torch.optim.AdamW(model.graph_encoder.pred_X.parameters(),
                                    **yaml_data["optimizer_X"])
    optimizer_E = torch.optim.AdamW(model.graph_encoder.pred_E.parameters(),
                                    **yaml_data["optimizer_E"])

    lr_scheduler_X = ReduceLROnPlateau(optimizer_X, mode='min', **yaml_data["lr_scheduler"])
    lr_scheduler_E = ReduceLROnPlateau(optimizer_E, mode='min', **yaml_data["lr_scheduler"])

    best_epoch_X = 0
    best_state_dict_X = deepcopy(model.graph_encoder.pred_X.state_dict())
    best_val_nll_X = float('inf')
    best_log_p_0_X = float('inf')
    best_denoise_match_X = float('inf')

    best_epoch_E = 0
    best_state_dict_E = deepcopy(model.graph_encoder.pred_E.state_dict())
    best_val_nll_E = float('inf')
    best_log_p_0_E = float('inf')
    best_denoise_match_E = float('inf')

    # Create the directory for saving model checkpoints.
    model_cpt_dir = f"{args.dataset}_cpts"
    os.makedirs(model_cpt_dir, exist_ok=True)

    num_patient_epochs = 0
    for epoch in range(train_config["num_epochs"]):
        model.train()

        for batch_edge_index in tqdm(data_loader):
            batch_edge_index = batch_edge_index.to(device)
            # (B), (B)
            batch_dst, batch_src = batch_edge_index.T
            loss_X, loss_E = model.log_p_t(X_one_hot_3d,
                                           E_one_hot,
                                           Y,
                                           batch_src,
                                           batch_dst,
                                           E_one_hot[batch_dst, batch_src])
            loss = loss_X + loss_E

            optimizer_X.zero_grad()
            optimizer_E.zero_grad()

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.graph_encoder.pred_X.parameters(), train_config["max_grad_norm"])
            nn.utils.clip_grad_norm_(
                model.graph_encoder.pred_E.parameters(), train_config["max_grad_norm"])

            optimizer_X.step()
            optimizer_E.step()

            wandb.log({"train/loss_X": loss_X.item(),
                       "train/loss_E": loss_E.item()})

        if (epoch + 1) % train_config["val_every_epochs"] != 0:
            continue

        model.eval()

        num_patient_epochs += 1

        denoise_match_E = []
        denoise_match_X = []
        log_p_0_E = []
        log_p_0_X = []
        for batch_edge_index in tqdm(val_data_loader):
            batch_dst, batch_src = batch_edge_index.T

            batch_denoise_match_E, batch_denoise_match_X,\
                batch_log_p_0_E, batch_log_p_0_X = model.val_step(
                    X_one_hot_3d,
                    E_one_hot,
                    Y,
                    batch_src,
                    batch_dst,
                    E_one_hot[batch_dst, batch_src])

            denoise_match_E.append(batch_denoise_match_E)
            denoise_match_X.append(batch_denoise_match_X)
            log_p_0_E.append(batch_log_p_0_E)
            log_p_0_X.append(batch_log_p_0_X)

        denoise_match_E = np.mean(denoise_match_E)
        denoise_match_X = np.mean(denoise_match_X)
        log_p_0_E = np.mean(log_p_0_E)
        log_p_0_X = np.mean(log_p_0_X)

        val_X = denoise_match_X + log_p_0_X
        val_E = denoise_match_E + log_p_0_E

        to_save_cpt = False
        if val_X < best_val_nll_X:
            best_val_nll_X = val_X
            best_epoch_X = epoch
            best_state_dict_X = deepcopy(model.graph_encoder.pred_X.state_dict())
            to_save_cpt = True

        if val_E < best_val_nll_E:
            best_val_nll_E = val_E
            best_epoch_E = epoch
            best_state_dict_E = deepcopy(model.graph_encoder.pred_E.state_dict())
            to_save_cpt = True

        if to_save_cpt:
            best_val_nll = best_val_nll_X + best_val_nll_E
            torch.save({
                "dataset": args.dataset,
                "train_yaml_data": yaml_data,
                "best_val_nll": best_val_nll,
                "best_epoch_X": best_epoch_X,
                "best_epoch_E": best_epoch_E,
                "pred_X_state_dict": best_state_dict_X,
                "pred_E_state_dict": best_state_dict_E
            }, f"{model_cpt_dir}/{model_name}_T{T}.pth")
            print('model saved')

        if log_p_0_X < best_log_p_0_X:
            best_log_p_0_X = log_p_0_X
            num_patient_epochs = 0

        if denoise_match_X < best_denoise_match_X:
            best_denoise_match_X = denoise_match_X
            num_patient_epochs = 0

        if log_p_0_E < best_log_p_0_E:
            best_log_p_0_E = log_p_0_E
            num_patient_epochs = 0

        if denoise_match_E < best_denoise_match_E:
            best_denoise_match_E = denoise_match_E
            num_patient_epochs = 0

        wandb.log({"epoch": epoch,
                   "val/denoise_match_X": denoise_match_X,
                   "val/denoise_match_E": denoise_match_E,
                   "val/log_p_0_X": log_p_0_X,
                   "val/log_p_0_E": log_p_0_E,
                   "val/best_log_p_0_X": best_log_p_0_X,
                   "val/best_denoise_match_X": best_denoise_match_X,
                   "val/best_log_p_0_E": best_log_p_0_E,
                   "val/best_denoise_match_E": best_denoise_match_E,
                   "val/best_val_X": best_val_nll_X,
                   "val/best_val_E": best_val_nll_E,
                   "val/best_val_nll": best_val_nll})

        print("Epoch {} | best val X nll {:.7f} | best val E nll {:.7f} | patience {}/{}".format(
            epoch, best_val_nll_X, best_val_nll_E, num_patient_epochs, train_config["patient_epochs"]))

        if num_patient_epochs == train_config["patient_epochs"]:
            break

        lr_scheduler_X.step(log_p_0_X)
        lr_scheduler_E.step(log_p_0_E)

    wandb.finish()


from sacred import Experiment
import seml

ex = Experiment()
seml.setup_logger(ex)

@ex.config
def config():
    seed = 42
    dataset = 'cora'


@ex.automain
def run(seed, model, dataset, _run, _log):  

    warnings.filterwarnings("ignore")


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'Using device {device}')
        
    train_diffuser(dataset=dataset)
