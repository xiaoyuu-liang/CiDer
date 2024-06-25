from sacred import Experiment
import seml


ex = Experiment()
seml.setup_logger(ex)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(
            db_collection, overwrite=overwrite))

    # default params
    dataset = 'cora'
    n_per_class = 20
    seed = 42

    patience = 50
    max_epochs = 3000
    lr = 1e-3
    weight_decay = 1e-3

    model = 'gcn'
    n_hidden = 64
    p_dropout = 0.5

    pf_plus_adj = 0.00
    pf_minus_adj = 0.00

    pf_plus_att = 0.01
    pf_minus_att = 0.65

    n_samples_train = 1
    batch_size_train = 1

    n_samples_pre_eval = 10
    n_samples_eval = 10000
    batch_size_eval = 10

    mean_softmax = False
    conf_alpha = 0.01
    early_stopping = True

    save_dir = 'rand_gnn_checkpoints'


@ex.automain
def run(_config, dataset, n_per_class, seed,
        patience, max_epochs, lr, weight_decay, model, n_hidden, p_dropout,
        pf_plus_adj, pf_plus_att, pf_minus_adj, pf_minus_att, conf_alpha,
        n_samples_train, n_samples_pre_eval, n_samples_eval, mean_softmax, early_stopping,
        batch_size_train, batch_size_eval, save_dir,
        ):
    import warnings
    import os
    warnings.filterwarnings('ignore')

    import numpy as np
    import torch
    from src.general_utils import load_and_standardize, get_train_val_test_split, binarize_labels
    from src.model.classifier import GCN, GAT, APPNP, SAGE
    from src.model.sparse_randomizer.training import train_gnn, train_pytorch
    from src.model.sparse_randomizer.prediction import predict_smooth_gnn, predict_smooth_pytorch
    from src.model.sparse_randomizer.cert import binary_certificate, joint_binary_certificate, minimize
    from src.model.sparse_randomizer.utils import (accuracy_majority, sample_batch_pyg)
    from torch_geometric.data import DataLoader as PyGDataLoader
    from src.general_utils import save_cetrificate
    print(_config)

    try:
        # os.makedirs('checkpoints')
        os.makedirs(save_dir)
    except OSError:
        pass

    try:
        # os.makedirs('checkpoints/' + args.general.name)
        os.makedirs(f'{save_dir}/{model}_{dataset}')
    except OSError:
        pass
    save_name = f'{save_dir}/{model}_{dataset}/X[{pf_plus_att}-{pf_minus_att}]_E[{pf_plus_adj}-{pf_minus_adj}].pt'

    sample_config = {
        'n_samples': n_samples_train,
        'pf_plus_adj': pf_plus_adj,
        'pf_plus_att': pf_plus_att,
        'pf_minus_adj': pf_minus_adj,
        'pf_minus_att': pf_minus_att,
    }

    # if we need to sample at least once and at least one flip probability is non-zero
    if n_samples_train > 0 and (pf_plus_adj+pf_plus_att+pf_minus_adj+pf_minus_att > 0):
        sample_config_train = sample_config
        sample_config_train['mean_softmax'] = mean_softmax
    else:
        sample_config_train = None
    sample_config_eval = sample_config.copy()
    sample_config_eval['n_samples'] = n_samples_eval
    
    sample_config_pre_eval = sample_config.copy()
    sample_config_pre_eval['n_samples'] = n_samples_pre_eval

    datafile = f'data/{dataset}.npz'
    graph = load_and_standardize(datafile)

    edge_idx = torch.LongTensor(
        np.stack(graph.adj_matrix.nonzero())).cuda()
    attr_idx = torch.LongTensor(
        np.stack(graph.attr_matrix.nonzero())).cuda()
    labels = torch.LongTensor(graph.labels).cuda()

    n, d = graph.attr_matrix.shape
    nc = graph.labels.max() + 1

    idx = {}
    bi_labels =  binarize_labels(graph.labels)
    idx['train'], idx['val'], idx['test'] = get_train_val_test_split(random_state=np.random.RandomState(seed),
                                                                     labels=bi_labels,
                                                                     train_examples_per_class=n_per_class,
                                                                     val_examples_per_class=n_per_class,
                                                                     test_examples_per_class=None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model.lower() == 'gcn':
        model = GCN(nfeat=graph.num_node_attr, nlayers=1, nhid=16, nclass=graph.num_classes, device=device)
    elif model.lower() == 'gat':
        model = GAT(nfeat=graph.num_node_attr, nhid=2, heads=8, nclass=graph.num_classes, device=device)
    elif model.lower() == 'appnp':
        model = APPNP(nfeat=graph.num_node_attr, nhid=16, K=8, alpha=0.15, nclass=graph.num_classes, device=device)
    elif model.lower() == 'sage':
        model = SAGE(nfeat=graph.num_node_attr, nhid=16, nclass=graph.num_classes, device=device)
    model.to(device)

    print(f'nodes: {n}, features: {d}, classes: {nc}')
    trace_val = train_gnn(model=model, edge_idx=edge_idx, attr_idx=attr_idx, labels=labels, n=n, d=d, nc=nc,
                          idx_train=idx['train'], idx_val=idx['val'], lr=lr, weight_decay=weight_decay,
                          patience=patience, max_epochs=max_epochs, display_step=10,
                          sample_config=sample_config_train,
                          batch_size=batch_size_train, early_stopping=early_stopping)

    pre_votes = predict_smooth_gnn(attr_idx=attr_idx, edge_idx=edge_idx,
                                   sample_config=sample_config_pre_eval,
                                   model=model, n=n, d=d, nc=nc,
                                   batch_size=batch_size_eval)

    votes = predict_smooth_gnn(attr_idx=attr_idx, edge_idx=edge_idx,
                               sample_config=sample_config_eval,
                               model=model, n=n, d=d, nc=nc,
                               batch_size=batch_size_eval)

    acc_clean = {}
    for split_name in ['train', 'val', 'test']:
        acc_clean[split_name] = accuracy_majority(votes=pre_votes, labels=graph.labels, idx=idx[split_name])
    acc_majority = {}
    for split_name in ['train', 'val', 'test']:
        acc_majority[split_name] = accuracy_majority(votes=votes, labels=graph.labels, idx=idx[split_name])

    votes_max = votes.max(1)[idx['test']]
    correct = votes.argmax(1)[idx['test']] == graph.labels[idx['test']]

    agreement = (votes.argmax(1) == pre_votes.argmax(1)).mean() 

    # we are perturbing ONLY the ATTRIBUTES
    if pf_plus_adj == 0 and pf_minus_adj == 0:
        print('Just ATT')
        grid_base, grid_lower, grid_upper = binary_certificate(
            votes=votes, pre_votes=pre_votes, n_samples=n_samples_eval, conf_alpha=conf_alpha,
            pf_plus=pf_plus_att, pf_minus=pf_minus_att)
    # we are perturbing ONLY the GRAPH
    elif pf_plus_att == 0 and pf_minus_att == 0:
        print('Just ADJ')
        grid_base, grid_lower, grid_upper = binary_certificate(
            votes=votes, pre_votes=pre_votes, n_samples=n_samples_eval, conf_alpha=conf_alpha,
            pf_plus=pf_plus_adj, pf_minus=pf_minus_adj)
    else:
        grid_base, grid_lower, grid_upper = joint_binary_certificate(
            votes=votes, pre_votes=pre_votes, n_samples=n_samples_eval, conf_alpha=conf_alpha,
            pf_plus_adj=pf_plus_adj, pf_minus_adj=pf_minus_adj,
            pf_plus_att=pf_plus_att, pf_minus_att=pf_minus_att)

    mean_max_ra_base = (grid_base > 0.5)[:, :, 0].argmin(1).mean()
    mean_max_rd_base = (grid_base > 0.5)[:, 0, :].argmin(1).mean()
    mean_max_ra_loup = (grid_lower >= grid_upper)[:, :, 0].argmin(1).mean()
    mean_max_rd_loup = (grid_lower >= grid_upper)[:, 0, :].argmin(1).mean()

    run_id = _config['overwrite']
    db_collection = _config['db_collection']
    
    # torch.save(model.state_dict(), save_name)
    # print(f'Saved model to {save_name}')

    binary_class_cert = (grid_base > 0.5)[idx['test']].T
    multi_class_cert = (grid_lower > grid_upper)[idx['test']].T

    # the returned result will be written into the database
    results = {
        'clean_acc': acc_clean['test'],
        'majority_acc': acc_majority['test'],
        'correct': correct.tolist(),
        "binary": {
            "ratios": minimize(binary_class_cert.mean(-1).T),
            "cert_acc": minimize((correct * binary_class_cert).mean(-1).T)
        },
        "multiclass": {
            "ratios": minimize(multi_class_cert.mean(-1).T),
            "cert_acc": minimize((correct * multi_class_cert).mean(-1).T)
        }
    }
    
    hparams = {
        'classifier': model.__class__.__name__.lower(),
        'smoothing_config': {
            'p': 1,
            'p_plus_adj': pf_plus_adj,
            'p_plus': pf_plus_att,
            'p_minus_adj': pf_minus_adj,
            'p_minus': pf_minus_att,
        },
    }
    save_cetrificate(results, dataset, hparams, f"{save_dir}/{hparams['classifier']}_{dataset}")
    return {k: results[k] for k in ('clean_acc', 'majority_acc')}