import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import wandb
import os

from src.model.diffusion.noise_schedule import PredefinedNoiseScheduleDiscrete, MarginalUniformTransition
from src.model.diffusion.transformer_model import GraphTransformer
from src.model.diffusion import utils
from src.model.diffusion import diffusion_utils


class GraphJointDiffuser(pl.LightningModule):
    def __init__(self, cfg, dataset_infos):
        super().__init__()
        
        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps

        self.Xdim = input_dims['X']
        self.Xcdim = input_dims['Xc']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Xcdim_output = output_dims['Xc']   
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps)
        
        self.model = GraphTransformer(n_layers=cfg.model.n_layers,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                      hidden_dims=cfg.model.hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU())
        
        if cfg.model.transition == 'marginal':
            x_marginals = self.dataset_info.node_attrs
            e_marginals = self.dataset_info.edge_types

            print(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")
            self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                              y_classes=self.ydim_output)
    

    def training_step(self, data, i):
        if data.edge_index.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        # pred = self.forward(noisy_data, node_mask)

        loss = 0
        import sys
        sys.exit()

        return {'loss': loss}
            

    def configure_optimizers(self):
        print('Using AdamW optimizer')
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                 weight_decay=self.cfg.train.weight_decay)
    
    def on_fit_start(self) -> None:
        print("on fit staring")
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        self.print("Size of the input features", self.Xdim, self.Xcdim, self.Edim, self.ydim)
        self.print("Size of the output features", self.Xdim_output, self.Xcdim_output, self.Edim_output, self.ydim_output)
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)
    

    def validation_step(self, data, i):
        pass


    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx, dx_c, dx_c) (bs, de, de)
        assert (abs(Qtb.X.sum(dim=3) - 1.) < 1e-4).all(), Qtb.X.sum(dim=3) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probX = (X.permute(0, 2, 1, 3) @ Qtb.X).permute(0, 2, 1, 3)  # (bs, n, dx, dx_c)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de)
        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xcdim)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)
        print(f"X shape {X_t.shape}, E shape {E_t.shape}")

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data
    

    def forward(self, noisy_data, node_mask):
        X = noisy_data['X_t'].float()
        E = noisy_data['E_t'].float()
        y = noisy_data['y_t'].float()
        return self.model(X, E, y, node_mask)