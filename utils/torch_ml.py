"""utility functions for general machine learning purposes in Pytorch"""
import torch
import torch_geometric
import networkx as nx
from torch_geometric.utils.convert import from_networkx

# Cache
DEVICE_CACHE = None


def get_device():
    """Return the used device (either cuda or cpu) by pytorch

    Returns
    -------
    torch.device
        Can be device(cuda) or device(cpu)
    """
    global DEVICE_CACHE
    if device_cache is None:
        device_cache = torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu")
        #device_cache = torch.device("cpu")
    return DEVICE_CACHE

def to_pyg_data(graph:nx.Graph) -> torch_geometric.data.Data:
    """Transform networkx Graph into torch_geometric Data

    Parameters
    ----------
    graph : nx.Graph
        A graph object to be adaptated 

    Returns
    -------
    torch_geometric.data.Data
    """
    pyg_data = from_networkx(graph)
    pyg_data.num_nodes = graph.number_of_nodes()
    return pyg_data