def get_marginal(graph):
    """
    Get the marginal distribution of the graph.

    Parameters
    ----------
    graph: SparseGraph The graph to get the marginal distribution from.

    Returns
    -------
    attr_margin: torch.Tensor of shape (F, 2) 
        The marginal distribution of X.
    label_margin: torch.Tensor of shape (|Y|) 
        The marginal distribution of Y.
    adj_margin: torch.Tensor of shape (2) 
        The marginal distribution of E.
    """
    attr_one_hot, adj_one_hot = get_one_hot(graph)
        
    # (F, 2)
    attr_one_hot_count = attr_one_hot.sum(dim=1)
    attr_margin = attr_one_hot_count / attr_one_hot_count.sum(dim=1, keepdim=True)

    label_one_hot = F.one_hot(torch.LongTensor(graph.labels), graph.num_classes)
    label_sum = label_one_hot.sum(dim=0)
    label_margin = label_sum / label_sum.sum()

    adj_one_hot_count = adj_one_hot.sum(dim=0).sum(dim=0)
    adj_margin = adj_one_hot_count / adj_one_hot_count.sum()


    return attr_margin, label_margin, adj_margin
    

def get_one_hot(graph):
    """
    Get the one-hot encoding of the graph.

    Parameters
    ----------
    graph: SparseGraph The graph to get the one-hot encoding from.

    Returns
    -------
    attr_one_hot: torch.Tensor of shape (F, N, 2) 
        The one-hot encoding of attr matrix.
        attr_one_hot[:, :, 0] = 1 <=> attr_matrix[i] = 0
        attr_one_hot[:, :, 1] = 1 <=> attr_matrix[i] = 1
    adj_one_hot: torch.Tensor of shape (N, N, 2)
        The one-hot encoding of adj matrix.
        adj_one_hot[:, :, 0] = 1 <=> adj_matrix[i, j] = 0
        adj_one_hot[:, :, 1] = 1 <=> adj_matrix[i, j] = 1
    """
    if sp.issparse(graph.attr_matrix):
        X = torch.LongTensor(graph.attr_matrix.todense())
    else:
        X = torch.LongTensor(graph.attr_matrix)
    
    if sp.issparse(graph.adj_matrix):
        A = torch.LongTensor(graph.adj_matrix.todense())
    else:
        A = torch.LongTensor(graph.adj_matrix)

    attr_one_hot_list = []
    for f in range(graph.num_node_attr):
        # (N, 2)
        attr_f_one_hot = F.one_hot(X[:, f], num_classes=2)
        attr_one_hot_list.append(attr_f_one_hot)
    # (F, N, 2)
    attr_one_hot = torch.stack(attr_one_hot_list, dim=0).float()

    # (N, N, 2)
    adj_one_hot = F.one_hot(A).float()

    return attr_one_hot, adj_one_hot
