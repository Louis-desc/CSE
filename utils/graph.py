"""Utility functions for graphs"""
from typing import Tuple

import random

from torch_geometric import data as pyg_data
from torch_geometric.utils.convert import to_networkx
import networkx as nx


def k_nodes_walk(graph:pyg_data.Data, start_node:int, k:int)-> nx.Graph:
    """Returns a networkX graph that represent a random walk from `start_node` and with `k` nodes. 

    Parameters
    ----------
    graph : pyg_data.Data
        target graph to sample a walk from
    start_node : int
        starting node of the walk
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