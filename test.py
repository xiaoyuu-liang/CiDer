import torch
import random
import numpy as np
import argparse
import warnings
import logging
import hydra
import os
import psutil
from omegaconf import DictConfig
from pympler import asizeof
from memory_profiler import profile

import torch.nn.functional as F
from pytorch_lightning import Trainer

from src import sparse_graph
from src.model.diffusion import utils
from src.model.diffusion import diffusion_utils
from src.model.diffusion.attributed_dataset import AttributedGraphDataModule, AttributedDatasetInfos
from src.model.diffusion.extra_features import DummyExtraFeatures
from src.model.diffusion.train_metrics import TrainAbstractMetricsDiscrete
from src.model.diffusion.diffusion_model import GraphJointDiffuser


def get_resume(cfg, model_kwargs):
    """ Resumes a run. It loads previous config without allowing to update keys (used for testing). """
    saved_cfg = cfg.copy()
    name = cfg.general.name + '_resume'
    resume = cfg.general.test_only

    model = GraphJointDiffuser.load_from_checkpoint(resume, **model_kwargs)
    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """ Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    saved_cfg = cfg.copy()
    # Fetch path to this file to get base path
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split('outputs')[0]

    resume_path = os.path.join(root_dir, cfg.general.resume)

    model = GraphJointDiffuser.load_from_checkpoint(resume_path, **model_kwargs)
    new_cfg = model.cfg

    for category in cfg:
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + '_resume'

    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model


@hydra.main(version_base='1.3', config_path='configs', config_name='config')
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]

    datamodule = AttributedGraphDataModule(cfg)
    dataset_infos = AttributedDatasetInfos(datamodule, dataset_config)
    extra_features = DummyExtraFeatures()
    domain_features = DummyExtraFeatures()
    dataset_infos.compute_input_output_dims(datamodule=datamodule, 
                                            extra_features=extra_features, domain_features=domain_features)

    train_metrics = TrainAbstractMetricsDiscrete()
    sampling_metrics = None
    model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                    'sampling_metrics': sampling_metrics,
                    'extra_features': extra_features, 'domain_features': domain_features}  

    if cfg.general.test_only:
        # When testing, previous configuration is fully loaded
        cfg, _ = get_resume(cfg, model_kwargs)
        os.chdir(cfg.general.test_only.split('checkpoints')[0])
    elif cfg.general.resume is not None:
        # When resuming, we can override some parts of previous configuration
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)
        os.chdir(cfg.general.resume.split('checkpoints')[0])

    utils.create_folders(cfg)
    
    callbacks = []
    name = cfg.general.name
    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()

    model = GraphJointDiffuser(cfg, **model_kwargs)
    trainer = Trainer(accelerator='gpu' if use_gpu else 'cpu',
                      devices=cfg.general.gpus if use_gpu else 1,
                      max_epochs=10)
    print(f"Training {name}")
    
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)


        
    

if __name__ == '__main__':
    main()