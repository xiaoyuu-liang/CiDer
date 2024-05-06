import numpy as np
import pydantic
import random
import torch
import yaml

from typing import Optional

# pydantic allows checking field types when loading configuration files
class MetaDataYaml(pydantic.BaseModel):
    variant: str

class GNNXYaml(pydantic.BaseModel):
    hidden_t: int
    hidden_X: int
    hidden_Y: int
    num_gnn_layers: int
    dropout: float

class GNNEYaml(pydantic.BaseModel):
    hidden_t: int
    hidden_X: int
    hidden_Y: int
    hidden_E: int
    num_gnn_layers: int
    dropout: float

class DiffusionYaml(pydantic.BaseModel):
    T: int

class OptimizerYaml(pydantic.BaseModel):
    lr: float
    weight_decay: Optional[float] = 0.
    amsgrad: Optional[bool] = False

class LRSchedulerYaml(pydantic.BaseModel):
    factor: float
    patience: int
    verbose: bool

class TrainYaml(pydantic.BaseModel):
    num_epochs: int
    val_every_epochs: int
    patient_epochs: int
    max_grad_norm: Optional[float] = None
    batch_size: int
    val_batch_size: int

class DiffuserYaml(pydantic.BaseModel):
    meta_data: MetaDataYaml
    gnn_X: GNNXYaml
    gnn_E: GNNEYaml
    diffusion: DiffusionYaml
    optimizer_X: OptimizerYaml
    optimizer_E: OptimizerYaml
    lr_scheduler: LRSchedulerYaml
    train: TrainYaml

class MLPXYaml(pydantic.BaseModel):
    hidden_t: int
    hidden_X: int
    hidden_Y: int
    num_mlp_layers: int
    dropout: float


def load_train_yaml(data_name):
    with open(f"configs/{data_name}.yaml") as f:
        yaml_data = yaml.load(f, Loader=yaml.loader.SafeLoader)

    return DiffuserYaml(**yaml_data).model_dump()

