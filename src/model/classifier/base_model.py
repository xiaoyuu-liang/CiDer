import torch
import logging

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from . import utils


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()   
        pass

    def fit(self, data, train_iters=1000, initialize=True, verbose=True, patience=100, **kwargs):
        if initialize:
            self.initialize()

        self.data = data.to(self.device)
        # By default, it is trained with early stopping on validation
        self.train_with_early_stopping(train_iters, patience, verbose)

    def finetune(self, edge_index, edge_weight, feat=None, train_iters=10, verbose=True):
        if verbose:
            print(f'=== finetuning {self.name} model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        labels = self.data.y
        if feat is None:
            x = self.data.x
        else:
            x = feat
        train_mask, val_mask = self.data.train_mask, self.data.val_mask
        best_loss_val = 100
        best_acc_val = 0
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(x, edge_index, edge_weight)
            loss_train = F.nll_loss(output[train_mask], labels[train_mask])
            loss_train.backward()
            optimizer.step()

            self.eval()
            with torch.no_grad():
                output = self.forward(x, edge_index)
            loss_val = F.nll_loss(output[val_mask], labels[val_mask])
            acc_val = utils.accuracy(output[val_mask], labels[val_mask])

            if verbose and i % 10 == 0:
                logging.info(f'Epoch {i}: training loss: {loss_train.item():.5f}, validation loss: {loss_val.item():.5f}, validation acc: {acc_val.item():.5f}')

            # if best_loss_val > loss_val:
            #     best_loss_val = loss_val
            #     best_output = output
            #     weights = deepcopy(self.state_dict())

            if best_acc_val < acc_val:
                best_acc_val = acc_val
                best_output = output
                weights = deepcopy(self.state_dict())

        logging.info('best_acc_val:', best_acc_val.item())
        self.load_state_dict(weights)
        return best_output


    def _fit_with_val(self, pyg_data, train_iters=1000, initialize=True, verbose=True, **kwargs):
        if initialize:
            self.initialize()

        # self.data = pyg_data[0].to(self.device)
        self.data = pyg_data.to(self.device)
        if verbose:
            print(f'=== training {self.name} model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        labels = self.data.y
        train_mask, val_mask = self.data.train_mask, self.data.val_mask

        x, edge_index = self.data.x, self.data.edge_index
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(x, edge_index)
            loss_train = F.nll_loss(output[train_mask+val_mask], labels[train_mask+val_mask])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                logging.info(f'Epoch {i}: training loss: {loss_train.item():.5f}')


    def fit_with_val(self, pyg_data, train_iters=1000, initialize=True, patience=100, verbose=True, **kwargs):
        if initialize:
            self.initialize()

        self.data = pyg_data.to(self.device)
        self.data.train_mask = self.data.train_mask + self.data.val1_mask
        self.data.val_mask = self.data.val2_mask
        self.train_with_early_stopping(train_iters, patience, verbose)

    def train_with_early_stopping(self, train_iters, patience, verbose):
        """
        early stopping based on the validation loss
        """
        
        if verbose:
            print(f'=== training {self.name} model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        labels = self.data.y
        train_mask, val_mask = self.data.train_mask, self.data.val_mask

        early_stopping = patience
        best_loss_val = 100
        best_acc_val = 0
        best_epoch = 0

        x, edge_index = self.data.x, self.data.edge_index
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            
            output = self.forward(x, edge_index)

            loss_train = F.nll_loss(output[train_mask], labels[train_mask])
            loss_train.backward()
            optimizer.step()

            self.eval()
            output = self.forward(x, edge_index)
            loss_val = F.nll_loss(output[val_mask], labels[val_mask])
            acc_val = utils.accuracy(output[val_mask], labels[val_mask])

            if verbose and i % 10 == 0:
                logging.info(f'Epoch {i}: training loss: {loss_train.item():.5f}, validation loss: {loss_val.item():.5f}, validation acc: {acc_val.item():.5f}')

            # if best_loss_val > loss_val:
            #     best_loss_val = loss_val
            #     self.output = output
            #     weights = deepcopy(self.state_dict())
            #     patience = early_stopping
            #     best_epoch = i
            # else:
            #     patience -= 1

            if best_acc_val < acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
                best_epoch = i
            else:
                patience -= 1

            if i > early_stopping and patience <= 0:
                break

        if verbose:
            # print('=== early stopping at {0}, loss_val = {1} ==='.format(best_epoch, best_loss_val) )
            print('=== early stopping at {0}, acc_val = {1} ==='.format(best_epoch, best_acc_val) )
        self.load_state_dict(weights)

    def test(self):
        """
        Evaluate model performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        test_mask = self.data.test_mask
        labels = self.data.y
        output = self.forward(self.data.x, self.data.edge_index)
        # output = self.output
        loss_test = F.nll_loss(output[test_mask], labels[test_mask])
        acc_test = utils.accuracy(output[test_mask], labels[test_mask])
        logging.info(f"Test set results: loss= {loss_test.item():.5f} accuracy= {acc_test.item():.5f}")
        return acc_test.item()

    def predict(self, x=None, edge_index=None, edge_weight=None):
        """
        Returns
        -------
        torch.FloatTensor
            output (log probabilities)
        """
        self.eval()
        if x is None or edge_index is None:
            x, edge_index = self.data.x, self.data.edge_index
        return self.forward(x, edge_index, edge_weight)

    def _ensure_contiguousness(self,
                               x,
                               edge_idx,
                               edge_weight):
        if not x.is_sparse:
            x = x.contiguous()
        if hasattr(edge_idx, 'contiguous'):
            edge_idx = edge_idx.contiguous()
        if edge_weight is not None:
            edge_weight = edge_weight.contiguous()
        return x, edge_idx, edge_weight



