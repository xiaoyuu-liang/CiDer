import torch
torch.cuda.empty_cache()
# print(torch.multiprocessing.get_start_method())
# torch.multiprocessing.set_start_method('spawn')
# torch.set_float32_matmul_precision('medium' | 'high')
import random
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import argparse
import warnings
import logging
import hydra
import os
import pathlib

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning
warnings.filterwarnings("ignore", category=PossibleUserWarning)

from src import general_utils
from src.model.classifier import GCN, GAT, APPNP, SAGE
from src.model.diffusion import utils
from src.model.diffusion.attributed_dataset import AttributedGraphDataModule, AttributedDatasetInfos
from src.model.diffusion.extra_features import DummyExtraFeatures
from src.model.diffusion.train_metrics import TrainAbstractMetricsDiscrete
from src.model.diffusion.diffusion_model import GraphJointDiffuser
from src.model.diffusion.extra_features import DummyExtraFeatures


def get_resume(cfg, model_kwargs):
    """ Resumes a run. It loads previous config without allowing to update keys (used for testing). """
    saved_cfg = cfg.copy()
    # Fetch path to this file to get base path
    root_dir = os.path.dirname(os.path.realpath(__file__))

    resume_path = os.path.join(root_dir, cfg.general.test_only)

    model = GraphJointDiffuser.load_from_checkpoint(resume_path, **model_kwargs)
    new_cfg = model.cfg

    for category in cfg:
        if category == 'denoiser':
            continue
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + '/test_only'

    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """ Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    saved_cfg = cfg.copy()
    # Fetch path to this file to get base path
    root_dir = os.path.dirname(os.path.realpath(__file__))

    resume_path = os.path.join(root_dir, cfg.general.resume)

    model = GraphJointDiffuser.load_from_checkpoint(resume_path, **model_kwargs)
    new_cfg = model.cfg

    for category in cfg:
        if category == 'denoiser':
            continue
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + '/test_only'

    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model


@hydra.main(version_base='1.3', config_path='configs', config_name='config')
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]
    denoiser_config = cfg["denoiser"]

    datamodule = AttributedGraphDataModule(cfg)
    dataset_infos = AttributedDatasetInfos(datamodule, dataset_config)
    dataset_infos.compute_input_output_dims(datamodule=datamodule)

    train_metrics = TrainAbstractMetricsDiscrete()

    extra_features = DummyExtraFeatures()
    model_kwargs = {'dataset_infos': dataset_infos, 'extra_features': extra_features,
                    'train_metrics': train_metrics}

    if cfg.general.test_only:
        # When testing, previous configuration is fully loaded
        cfg, _ = get_resume(cfg, model_kwargs)
    elif cfg.general.resume is not None:
        # When resuming, we can override some parts of previous configuration
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)

    utils.create_folders(cfg)
    
    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{dataset_config['name']}/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val/epoch_NLL',
                                              save_top_k=5,
                                              mode='min',
                                              every_n_epochs=1)
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{dataset_config['name']}/{cfg.general.name}", filename='last', every_n_epochs=1)
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)

    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    name = cfg.general.name
    use_gpu = (isinstance(cfg.general.gpus, str) or cfg.general.gpus > 0) and torch.cuda.is_available()
    device = 'cuda:' + cfg.general.gpus if isinstance(cfg.general.gpus, str) else 'cuda:0' if cfg.general.gpus > 0 else 'cpu'
    print(f"Using device {device}")

    model = GraphJointDiffuser(cfg, **model_kwargs)
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      strategy="auto",  # Needed to load old checkpoints
                      accelerator='gpu' if use_gpu else 'cpu',
                      devices=cfg.general.gpus if use_gpu else 1,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      enable_progress_bar=cfg.train.progress_bar,
                      callbacks=callbacks,
                      log_every_n_steps=50 if name != 'debug' else 1,
                      logger = [])
    
    if not cfg.general.test_only: # train / finetune
        print(f"Training {name}")
        trainer.fit(model, 
                    train_dataloaders=datamodule.train_dataloader(),
                    val_dataloaders=datamodule.val_dataloader(), 
                    ckpt_path=cfg.general.resume)
        # trainer.test(model,
        #              dataloaders=datamodule.test_dataloader())
    elif cfg.general.evaluate_all_checkpoints: # evaluate all checkpoints
        directory = pathlib.Path(cfg.general.test_only).parents[0]
        print("Directory:", directory)
        files_list = os.listdir(directory)
        for file in files_list:
            if '.ckpt' in file:
                ckpt_path = os.path.join(directory, file)
                print("Loading checkpoint", ckpt_path)
                trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)
    else: # predict
        print('denoise with graph joint diffuser')
        _, model = get_resume(cfg, model_kwargs)
        
        classifier_name = denoiser_config.classifier
        nlayer = denoiser_config.classifier_nlayers
        nfeat = len(dataset_infos.node_attrs)
        nclass = len(dataset_infos.node_types)
        # Setup model
        if classifier_name == 'gcn':
            classifier = GCN(nfeat=nfeat, nlayers=nlayer, nhid=16, nclass=nclass, device=device)
        elif classifier_name == 'gat':
            classifier = GAT(nfeat=nfeat, nlayers=nlayer, nhid=2, heads=8, nclass=nclass, device=device)
        elif classifier_name == 'appnp':
            classifier = APPNP(nfeat=nfeat, nhid=16, K=8, alpha=0.15, nclass=nclass, device=device)
        elif classifier_name == 'sage':
            classifier = SAGE(nfeat=nfeat, nhid=16, nclass=nclass, device=device)
        classifier.to(device)
        # classifier.load_state_dict(torch.load(denoiser_config.classifier_path))

        classifier.eval()
        classifier.to(device)
        hparams = OmegaConf.to_container(denoiser_config)
        smoothing_config = model.compute_noise(t_X=denoiser_config.attr_noise_scale, t_E=denoiser_config.adj_noise_scale)   
        hparams['smoothing_config'] = smoothing_config
        # dict_to_save = model.denoised_smoothing(dataloader=datamodule.test_dataloader(),
        #                                         classifier=classifier,
        #                                         hparams=hparams)
        # general_utils.save_cetrificate(dict_to_save, dataset_config, hparams, f"checkpoints/{dataset_config['name']}/{cfg.general.name}")
        

if __name__ == '__main__':
    main()
