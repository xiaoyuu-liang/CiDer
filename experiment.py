import numpy as np
import torch

from src.model.randomizer.cert import *
from src.model.randomizer.datasets import *
from src.model.randomizer.utils import *

from src.model.classifier import GCN, GAT, APPNP, SAGE
from src.model.randomizer.create import *
from src.model.randomizer.training import *
from src.model.randomizer.prediction import *
from src.model.randomizer.smoothing import *


class Experiment():

    def run(self, hparams):
        results = {}
        dict_to_save = {}

        seeds = torch.load(hparams["fixed_random_seeds_path"])
        seeds = seeds[:hparams["num_seeds"]]
        for seed in seeds:
            dict_to_save[seed] = {}

            data = load_dataset(hparams, seed=seed)
            [A, X, y, n, d, nc, train, valid, test,
                idx_train, idx_valid, idx_test] = data
            data_train = prepare_graph_data(train, hparams['device'])
            data_valid = prepare_graph_data(valid, hparams['device'])
            data_test = prepare_graph_data(test, hparams['device'])

            set_random_seed(seed)
            model = create_gnn(hparams)

            training_data = (data_train, data_valid, idx_train, idx_valid)
            model = train_gnn_inductive(model, training_data, hparams)
            model.eval()

            if not hparams['protected']:
                acc = predict_unprotected_graphs(model, data_test, idx_test)
                dict_to_save[seed]["acc"] = acc
                continue

            pre_votes = smooth_graph_classifier(
                hparams, model, data_test, hparams["pre_n_samples"])
            votes = smooth_graph_classifier(
                hparams, model, data_test, hparams["n_samples"])

            pre_votes = pre_votes[idx_test]
            votes = votes[idx_test]
            y_hat = pre_votes.argmax(1)
            y = data_test.y.cpu()
            correct = (y_hat == y).numpy()
            clean_acc = correct.mean()
            y_majority = votes.argmax(1)
            majority_correct = (y_majority == y).numpy()
            majority_acc = majority_correct.mean()

            dict_to_save[seed] = certify(correct, votes, pre_votes, hparams)
            dict_to_save[seed]["clean_acc"] = clean_acc
            dict_to_save[seed]["majority_acc"] = majority_acc
            dict_to_save[seed]["correct"] = correct.tolist()

        if not hparams['protected']:
            accs = [dict_to_save[k]["acc"] for k in seeds]
            dict_to_save["acc"] = np.mean(accs), np.std(accs)
            return results, dict_to_save

        # AVG
        clean_accs = [dict_to_save[k]['clean_acc'] for k in seeds]
        dict_to_save['clean_acc'] = np.mean(clean_accs), np.std(clean_accs)

        majority_accs = [dict_to_save[k]['majority_acc'] for k in seeds]
        dict_to_save['majority_acc'] = np.mean(majority_accs), np.std(majority_accs)
        print(f'clean acc: {dict_to_save["clean_acc"]}, majority acc: {dict_to_save["majority_acc"]}')

        abstains = [dict_to_save[k]['abstain_binary'] for k in seeds]
        dict_to_save['abstain_binary'] = np.mean(abstains), np.std(abstains)

        abstains = [dict_to_save[k]['abstain_multiclass'] for k in seeds]
        averaged_result = np.mean(abstains), np.std(abstains)
        dict_to_save['abstain_multiclass'] = averaged_result

        smoothing_config = hparams['smoothing_config']
        smoothing_distribution = smoothing_config['smoothing_distribution']
        if smoothing_distribution in ["sparse", "hierarchical_sparse"]:
            dict_to_save['binary'] = {
                "ratios": avg_results(dict_to_save, "binary",
                                      "ratios", seeds),
                "cert_acc": avg_results(dict_to_save, "binary",
                                        "cert_acc", seeds),
            }
            dict_to_save['multiclass'] = {
                "ratios": avg_results(dict_to_save, "multiclass",
                                      "ratios", seeds),
                "cert_acc": avg_results(dict_to_save, "multiclass",
                                        "cert_acc", seeds),
            }

            # cleanup
            for seed in seeds:
                del dict_to_save[seed]['binary']
                del dict_to_save[seed]['multiclass']

        return results, dict_to_save
