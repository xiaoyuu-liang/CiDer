import numpy as np
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split
import torch.sparse as ts
import torch.nn.functional as F
import warnings


def accuracy(output, labels):
    """Return accuracy of output compared to labels.

    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels

    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def loss_acc(output, labels, targets, avg_loss=True):
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()[targets]
    loss = F.nll_loss(output[targets], labels[targets], reduction='mean' if avg_loss else 'none')

    if avg_loss:
        return loss, correct.sum() / len(targets)
    return loss, correct
    # correct = correct.sum()
    # return loss, correct / len(labels)