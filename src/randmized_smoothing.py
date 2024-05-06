import torch
import numpy as np
from typing import List

from torch_sparse import coalesce


def sparse_perturb_multiple(data_idx, pf_minus, pf_plus, n, m, undirected, nsamples, offset_both_idx):
    """
    Randomly flip bits.

    Parameters
    ----------
    data_idx: torch.Tensor [2, ?]
        The indices of the non-zero elements
    pf_minus: float, 0 <= p_plus <= 1
        The probability to flip a one to a zero
    pf_plus : float, 0 <= p_plus <= 1
        The probability to flip a zero to a one
    n : int
        The shape of the tensor
    m : int
        The shape of the tensor
    undirected : bool
        If true for every (i, j) also perturb (j, i)
    nsamples : int
        Number of perturbed samples
    offset_both_idx : bool
        Whether to offset both matrix indices (for adjacency matrix)

    Returns
    -------
    perturbed_data_idx: torch.Tensor [2, ?]
        The indices of the non-zero elements of multiple concatenated matrices
        after perturbation
    """
    if undirected:
        # select only one direction of the edges, ignore self loops
        data_idx = data_idx[:, data_idx[0] < data_idx[1]]

    idx_copies = copy_idx(data_idx, n, nsamples, offset_both_idx)
    w_existing = torch.ones_like(idx_copies[0])
    to_del = torch.cuda.BoolTensor(idx_copies.shape[1]).bernoulli_(pf_minus)
    w_existing[to_del] = 0

    if offset_both_idx:
        assert n == m
        nadd_persample_np = np.random.binomial(n * m, pf_plus, size=nsamples)  # 6x faster than PyTorch
        nadd_persample = torch.cuda.FloatTensor(nadd_persample_np)
        nadd_persample_with_repl = torch.round(torch.log(1 - nadd_persample / (n * m))
                                               / np.log(1 - 1 / (n * m))).long()
        nadd_with_repl = nadd_persample_with_repl.sum()
        to_add = data_idx.new_empty([2, nadd_with_repl])
        to_add[0].random_(n * m)
        to_add[1] = to_add[0] % m
        to_add[0] = to_add[0] // m
        to_add = offset_idx(to_add, nadd_persample_with_repl, m, [0, 1])
        if undirected:
            # select only one direction of the edges, ignore self loops
            to_add = to_add[:, to_add[0] < to_add[1]]
    else:
        nadd = np.random.binomial(nsamples * n * m, pf_plus)  # 6x faster than PyTorch
        nadd_with_repl = int(np.round(np.log(1 - nadd / (nsamples * n * m))
                                      / np.log(1 - 1 / (nsamples * n * m))))
        to_add = data_idx.new_empty([2, nadd_with_repl])
        to_add[0].random_(nsamples * n * m)
        to_add[1] = to_add[0] % m
        to_add[0] = to_add[0] // m

    w_added = torch.ones_like(to_add[0])

    if offset_both_idx:
        mb = nsamples * m
    else:
        mb = m

    # if an edge already exists but has been removed do not add it back
    # hence we coalesce with the min value
    joined, weights = coalesce(torch.cat((idx_copies, to_add), 1),
                               torch.cat((w_existing, w_added), 0),
                               nsamples * n, mb, 'min')

    per_data_idx = joined[:, weights > 0]

    if undirected:
        per_data_idx = torch.cat((per_data_idx, per_data_idx[[1, 0]]), 1)

    # Check that there are no off-diagonal edges
    # if offset_both_idx:
    #     batch0 = to_add[0] // n
    #     batch1 = to_add[1] // n
    #     assert torch.all(batch0 == batch1)

    return per_data_idx


def copy_idx(idx: torch.LongTensor, dim_size: int, ncopies: int, offset_both_idx: bool):
    idx_copies = idx.repeat(1, ncopies)

    offset = dim_size * torch.arange(ncopies, dtype=torch.long,
                                     device=idx.device)[:, None].expand(ncopies, idx.shape[1]).flatten()

    if offset_both_idx:
        idx_copies += offset[None, :]
    else:
        idx_copies[0] += offset

    return idx_copies


def offset_idx(idx_mat: torch.LongTensor, lens: torch.LongTensor, dim_size: int, indices: List[int] = [0]):
    offset = dim_size * torch.arange(len(lens), dtype=torch.long,
                                     device=idx_mat.device).repeat_interleave(lens, dim=0)

    idx_mat[indices, :] += offset[None, :]
    return idx_mat


def sample_multiple_graphs(pyg_graph, sample_config, nsamples):
    """
    Perturb the structure and node attributes.
    Randomization scheme from https://github.com/abojchevski/sparse_smoothing

    Parameters
    ----------
    graph: torch_geometric.data.Data
        The graph to perturb.
    sample_config: dict
        Configuration specifying the sampling probabilities
    nsamples : int
        Number of samples

    Returns
    -------
    list of torch_geometric.data.Data
        The perturbed graphs.
    """
    pf_plus_adj = sample_config.get('pf_plus_adj', 0)
    pf_plus_att = sample_config.get('pf_plus_att', 0)

    pf_minus_adj = sample_config.get('pf_minus_adj', 0)
    pf_minus_att = sample_config.get('pf_minus_att', 0)

    if pf_minus_att + pf_plus_att > 0:
        per_attr_idx = sparse_perturb_multiple(data_idx=pyg_graph.x, n=pyg_graph.num_nodes, m=pyg_graph.num_features, 
                                               undirected=False, pf_minus=pf_minus_att, pf_plus=pf_plus_att,
                                               nsamples=nsamples, offset_both_idx=False)
    else:
        per_attr_idx = copy_idx(idx=pyg_graph.x, dim_size=pyg_graph.num_nodes, ncopies=nsamples, offset_both_idx=False)

    if pf_minus_adj + pf_plus_adj > 0:
        per_edge_idx = sparse_perturb_multiple(data_idx=pyg_graph.edge_index, n=pyg_graph.num_nodes, m=pyg_graph.num_nodes, 
                                               undirected=True,pf_minus=pf_minus_adj, pf_plus=pf_plus_adj,
                                               nsamples=nsamples, offset_both_idx=True)
    else:
        per_edge_idx = copy_idx(idx=pyg_graph.edge_index, dim_size=pyg_graph.num_nodes, ncopies=nsamples, offset_both_idx=True)

    return per_attr_idx, per_edge_idx
