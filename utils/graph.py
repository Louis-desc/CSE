"""Utility functions for graphs"""
from typing import Tuple
from abc import ABC, abstractmethod

import random

from torch_geometric import data as pyg_data
from torch_geometric.utils.convert import to_networkx
import networkx as nx
import numpy as np

def k_nodes_walk(graph:pyg_data.Data, start_node:int, k:int)-> nx.Graph:
    """Returns a networkX subgraph induced by nodes that represent a random walk from `start_node` and with `k` nodes. 
    It means that every edeges between the nodes of the walk are included in the subgraph (even if they are not used during the random walk).

    Parameters
    ----------
    graph : pyg_data.Data
        target graph to sample a walk from
    start_node : int
        starting node for the walk
    k : int
        length of the walk (number of nodes)

    Returns
    -------
    nx.Graph
        A networkX graph of the walk
    """
    query_node = []
    frontier = [start_node]
    nx_graph = to_networkx(graph,to_undirected=True)
    assert k < graph.num_nodes
    for _ in range(k):                                        #Random walk of `k` hops chosen in every neighbors of the current query
        new_node = random.choice(frontier)                                  #Choosing a node among the neighbors of the graph
        query_node.append(new_node)                                         #Adding the new node to the walk
        frontier += list(nx_graph.neighbors(new_node))                      #Modifying the frontier of the new graph
        frontier = [node for node in frontier if node not in query_node]    #Deleting already encountered nodes

    assert len(query_node) == len(set(query_node))

    return nx_graph.subgraph(query_node)


def relabel_graph_by_int(graph:nx.Graph) -> Tuple[nx.Graph, dict]:
    """Relabel graph nodes by integers between 0 and number of nodes(-1)

    Parameters
    ----------
    graph : nx.Graph
        Graph to relabel

    Returns
    -------
    Tuple[nx.Graph, dict]
        Returns the permuted graph and the permutation dictionnary ({old_node_label : new_label})
    """
    nodes = list(graph.nodes)
    perm = {nodes[i] : i for i in range(graph.number_of_nodes())}
    relabeled_graph = nx.relabel_nodes(graph,perm)
    return relabeled_graph, perm


# --- Abstract Class for graphs ---

class Generator(ABC):
    r"""
    Abstract class of on the fly generator used in the dataset.
    It generates on the fly graphs, which will be fed into the model.
    """
    def __init__(self, sizes=None):
        self._sizes_list = sizes

    def _get_size(self, size=None):
        if size is None:
            return np.random.choice(
                self._sizes_list, size=1, replace=True
            )[0]
        else:
            return size

    def __next__(self) :
        return self.generate()

    def __iter__(self) :
        return self

    def __len__(self):
        return 1

    @abstractmethod
    def generate(self):
        r"""
        Overwrite in subclass. Generates and returns a 
        :class:`networkx.Graph` object

        Returns:
            :class:`networkx.Graph`: A networkx graph object.
        """