import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from . import utils

class MPNN(nn.Module):
    """
    Message Passing Neural Network (MPNN) model.
    """
    def __init__(self, input_dims: dict, n_mlp_layers: int, hidden_mlp_dims: dict, mlp_dropout: float,
                 n_gnn_layers: int, hidden_gnn_dims: dict, gnn_dropout: float, output_dims: dict,
                 act_fn_in: nn.ReLU, act_fn_out: nn.ReLU):
        super().__init__()

        self.attr_predictor = MLP(input_dims, n_mlp_layers, hidden_mlp_dims, output_dims, mlp_dropout, act_fn_in, act_fn_out)

        self.link_predictor = GNN(input_dims, n_gnn_layers, hidden_gnn_dims, output_dims, gnn_dropout, act_fn_in, act_fn_out)
    
    def forward(self, X, E, y, label, node_mask):
        bs, n, bx, bx_c = X.shape

        X = self.attr_predictor(X, y, label, node_mask)             # (bs, n, bx*bx_c)
        X = X.view(bs, n, bx, bx_c)                                 # (bs, n, bx, bx_c)

        E = self.link_predictor(X, E, y, label, node_mask)

        return utils.PlaceHolder(X=X, E=E, y=y).mask(node_mask)


class MLP(nn.Module):
    """
    Multi-layer Perceptron (MLP) model.
    """
    def __init__(self, input_dims: dict, n_layers: int, hidden_dims: dict, output_dims: dict,
                 dropout: float, act_fn_in: nn.ReLU, act_fn_out: nn.ReLU):
        super().__init__()
        
        self.mlp_in_X = nn.Sequential(nn.Linear(input_dims['X']*input_dims['Xc'], hidden_dims['X']), act_fn_in,
                                      nn.Linear(hidden_dims['X'], hidden_dims['X']), act_fn_in)
        self.mlp_in_y = nn.Sequential(nn.Linear(input_dims['y'], hidden_dims['y']), act_fn_in,
                                      nn.Linear(hidden_dims['y'], hidden_dims['y']), act_fn_in)
        self.emd_label = nn.Embedding(input_dims['label']+1, hidden_dims['label'])                  # add fake label

        self.mlp_layers = nn.ModuleList([MLPLayer(hidden_dims, act_fn_in, dropout) for _ in range(n_layers)])

        hidden_cat = (n_layers + 1) * (hidden_dims['X'] + hidden_dims['label']) + hidden_dims['y']
        self.mlp_out = nn.Sequential(nn.Linear(hidden_cat, hidden_cat), act_fn_out,
                                    nn.Linear(hidden_cat, output_dims['X']*output_dims['Xc']))

    def forward(self, X, y, label, node_mask):
        bs, n, bx, bx_c = X.shape
        X = X.view(bs, n, -1)                   # (bs, n, bx*bx_c)
        label = label + 1                       # (-1 ~ node_classes) -> (0 ~ node_classes+1)
        
        x_mask = node_mask.unsqueeze(-1)        # bs, n, 1 
        X = self.mlp_in_X(X) * x_mask

        y = self.mlp_in_y(y)
        _, hy = y.shape

        label = self.emd_label(label) * x_mask
        
        X_list = [X]
        label_list = [label]
        for layer in self.mlp_layers:
            X, label = layer(X, y, label)

            X = X * x_mask
            label = label * x_mask

            X_list.append(X)
            label_list.append(label)
        
        y_expand = y.unsqueeze(1).expand(bs, n, hy)
        X = torch.cat(X_list + label_list + [y_expand], dim=-1)

        X = self.mlp_out(X)

        return X


class MLPLayer(nn.Module):
    """
    Multi-layer Perceptron (MLP) layer.
    """
    def __init__(self, hidden_dims: dict, act_fn: nn.ReLU, dropout: float):
        super().__init__()
        
        self.update_X = nn.Sequential(nn.Linear(hidden_dims['X'] + hidden_dims['y'] + hidden_dims['label'], hidden_dims['X']), act_fn,
                                      nn.LayerNorm(hidden_dims['X']), nn.Dropout(dropout))
        
        self.update_label = nn.Sequential(nn.Linear(hidden_dims['label'], hidden_dims['label']), act_fn,
                                          nn.LayerNorm(hidden_dims['label']), nn.Dropout(dropout))
        
    def forward(self, X, y, label):
        bs, n, hx = X.shape
        _, hy = y.shape

        y_expand = y.unsqueeze(1).expand(bs, n, hy)
        X = torch.cat([X, label, y_expand], dim=-1)

        X = self.update_X(X)                    # (bs, n, hx)
        label = self.update_label(label)        # (bs, n, hl)
        return X, label


class GNN(nn.Module):
    """
    Graph Neural Network (GNN) model.
    """
    def __init__(self, input_dims: dict, n_layers: int, hidden_dims: dict, output_dims: dict,
                 dropout: float, act_fn_in: nn.ReLU, act_fn_out: nn.ReLU):
        super().__init__()

        self.mlp_in_X = nn.Sequential(nn.Linear(input_dims['X']*input_dims['Xc'], hidden_dims['X']), act_fn_in,
                                      nn.Linear(hidden_dims['X'], hidden_dims['X']), act_fn_in)
        self.mlp_in_y = nn.Sequential(nn.Linear(input_dims['y'], hidden_dims['y']), act_fn_in,
                                      nn.Linear(hidden_dims['y'], hidden_dims['y']), act_fn_in)
        self.emd_label = nn.Embedding(input_dims['label']+1, hidden_dims['label'])                  # add fake label

        self.gnn_layers = nn.ModuleList([GNNLayer(hidden_dims, act_fn_in, dropout) for _ in range(n_layers)])

        hidden_cat = (n_layers + 1) * (hidden_dims['X'] + hidden_dims['label']) + hidden_dims['y']
        self.gnn_out = nn.Sequential(nn.Linear(hidden_cat, hidden_cat), act_fn_out,
                                    nn.Linear(hidden_cat, hidden_dims['E']))
        
        self.mlp_out = nn.Sequential(nn.Linear(hidden_dims['E'], hidden_dims['E']), act_fn_out,
                                     nn.Linear(hidden_dims['E'], output_dims['E']))
    
    def forward(self, X, E, y, label, node_mask):
        bs, n, bx, bx_c = X.shape
        X = X.view(bs, n, -1)                   # (bs, n, bx*bx_c)
        label = label + 1                       # (-1 ~ node_classes) -> (0 ~ node_classes+1)
        
        x_mask = node_mask.unsqueeze(-1)        # bs, n, 1 
        X = self.mlp_in_X(X) * x_mask

        y = self.mlp_in_y(y)
        _, hy = y.shape

        label = self.emd_label(label)  * x_mask

        X_list = [X]
        label_list = [label]
        for layer in self.gnn_layers:
            X, label = layer(X, E, y, label, node_mask)
            print(X.shape, label.shape)
            import sys
            sys.exit()
            X = X * x_mask
            label = label * x_mask

            X_list.append(X)
            label_list.append(label)
        
        y_expand = y.unsqueeze(1).expand(bs, n, hy)
        X = torch.cat(X_list + label_list + [y_expand], dim=-1)

        E = self.gnn_out(X)
        # TODO: get edge features
        # # (|E|, hidden_E)
        # h = h[src] * h[dst]
        E = self.mlp_out(E)

        return E


class GNNLayer(nn.Module):
    """
    Graph Neural Network (GNN) layer.
    """
    def __init__(self, hidden_dims: dict, act_fn: nn.ReLU, dropout: float):
        super().__init__()

        self.aggr_X = GCNConv(hidden_dims['X'] + hidden_dims['label'], hidden_dims['X'])
        self.aggr_label = GCNConv(hidden_dims['label'], hidden_dims['label'])

        self.update_X = nn.Sequential(nn.Linear(hidden_dims['X'] + hidden_dims['y'] + hidden_dims['label'], hidden_dims['X']), act_fn,
                                      nn.LayerNorm(hidden_dims['X']), nn.Dropout(dropout))
        
        self.update_label = nn.Sequential(nn.Linear(hidden_dims['label'], hidden_dims['label']), act_fn,
                                          nn.LayerNorm(hidden_dims['label']), nn.Dropout(dropout))
        
    def forward(self, X, E, y, label, node_mask):
        bs, n, hx = X.shape
        _, hy = y.shape

        adj = E[..., 1]
        edge_index = torch.nonzero(adj).t()
        print(edge_index.shape)
        
        # TODO: reshape X, edge_index ?
        pyg_X, edge_index = utils.to_sparse(X, adj, node_mask)
        pyg_label, _ = utils.to_sparse(label, adj, node_mask)

        X = torch.cat([X, label], dim=-1)
        X = self.aggr_X(X, edge_index)
        label = self.aggr_label(label, edge_index)

        y_expand = y.unsqueeze(1).expand(bs, n, hy)
        X = torch.cat([X, label, y_expand], dim=-1)

        X = self.update_X(X)                    # (bs, n, hx)
        label = self.update_label(label)        # (bs, n, hl)
        return X, label