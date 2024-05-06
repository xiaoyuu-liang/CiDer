import seml.experiment
import torch
import random
import numpy as np
import argparse
import warnings
import logging
import hydra
import os
from omegaconf import DictConfig

from src import sparse_graph
from src import utils
from src.model.diffusion.attributed_dataset import AttributedGraphDataModule, AttributedDatasetInfos
from src.model.diffusion.extra_features import DummyExtraFeatures


@hydra.main(version_base='1.3', config_path='configs', config_name='config')
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]

    datamodule = AttributedGraphDataModule(cfg)
    dataset_infos = AttributedDatasetInfos(datamodule, dataset_config)
    extra_features = DummyExtraFeatures()
    domain_features = DummyExtraFeatures()
    dataset_infos.compute_input_output_dims(datamodule=datamodule, 
                                            extra_features=extra_features, domain_features=domain_features)
    print(f'input dims: {dataset_infos.input_dims}, output dims: {dataset_infos.output_dims}')

if __name__ == '__main__':
    main()