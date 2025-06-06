import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import APPNP as APPNPConv
from torch.nn import Linear
from .base_model import BaseModel


class APPNP(BaseModel):

    def __init__(self, nfeat, nhid, nclass, K=10, alpha=0.1, dropout=0.5, lr=0.01,
                with_bn=False, weight_decay=5e-4, with_bias=True, device=None):

        super(APPNP, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device


        self.lin1 = Linear(nfeat, nhid)
        if with_bn:
            self.bn1 = nn.BatchNorm1d(nhid)
            self.bn2 = nn.BatchNorm1d(nclass)

        self.lin2 = Linear(nhid, nclass)
        self.prop1 = APPNPConv(K, alpha)

        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.output = None
        self.best_model = None
        self.best_output = None
        self.name = 'APPNP'
        self.with_bn = with_bn

        self.token = nn.Parameter(torch.zeros(nfeat))
        nn.init.xavier_uniform_(self.token.unsqueeze(0))

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        if self.with_bn:
            x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        if self.with_bn:
            x = self.bn2(x)
        x = self.prop1(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


    def initialize(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        if self.with_bn:
            self.bn1.reset_parameters()
            self.bn2.reset_parameters()

