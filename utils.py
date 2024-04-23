import numpy as np
import scipy.sparse as sp
from .sparse_graph import SparseGraph


def load_and_standardize(file_name):
    """
    Run standardize() + make the attributes binary.

    Parameters
    ----------
    file_name
        Name of the file to load.
    Returns
    -------
    graph: SparseGraph
        The standardized graph

    """
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        if 'type' in loader:
            del loader['type']
        graph = SparseGraph.from_flat_dict(loader)
    
    graph.standardize()

    # binarize
    graph._flag_writeable(True)
    graph.adj_matrix[graph.adj_matrix != 0] = 1
    graph.attr_matrix[graph.attr_matrix != 0] = 1
    graph._flag_writeable(False)

    return graph
