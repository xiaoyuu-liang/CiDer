import numpy as np
import seml
import torch
from sacred import Experiment as SacredExperiment
import time
import yaml
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

from experiment import Experiment

def load_config_from_yaml(yaml_file_path):
    with open(yaml_file_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            config = None
    return config

ex = SacredExperiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)

yaml_data =  load_config_from_yaml('rand_configs/hierarchical.yaml')

@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(
            db_collection, overwrite=overwrite))

    conf = yaml_data.get('conf', {})
    hparams = yaml_data.get('hparams', {})


@ex.automain
def run(_config, conf: dict, hparams: dict):
    print(conf)
    print(hparams)

    start = time.time()
    experiment = Experiment()
    results, dict_to_save = experiment.run(hparams)
    end = time.time()
    print(f"time={end-start}s")
    results['time'] = end-start

    save_dir = conf["save_dir"]
    run_id = _config['overwrite']
    db_collection = _config['db_collection']

    if conf["save"]:
        arch = hparams['arch']
        dataset = hparams['dataset']
        p = hparams['smoothing_config']['p']
        p_plus = hparams['smoothing_config']['p_plus']
        p_minus = hparams['smoothing_config']['p_minus']

        print(f'saving to {save_dir}/{arch}_{dataset}_[{p}-{p_plus}-{p_minus}].pth')
        torch.save(dict_to_save, f'{save_dir}/{arch}_{dataset}_[{p}-{p_plus}-{p_minus}].pth')
    return results