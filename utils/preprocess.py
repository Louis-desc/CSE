"""utility functions for preprocessing"""
# --- Imports
import torch
import torch_geometric.utils as pyg_utils
import networkx as nx
import numpy as np
from utils.torch_ml import get_device


# --- Not for import functions

def _one_hot_tensor(list_scalars, one_hot_dim=1):
    if not isinstance(list_scalars, list) and not list_scalars.ndim == 1:
        raise ValueError("input to _one_hot_tensor must be 1-D list")
    vals = torch.LongTensor(list_scalars).view(-1,1)
    vals = vals - min(vals)
    vals = torch.min(vals, torch.tensor(one_hot_dim - 1))
    vals = torch.max(vals, torch.tensor(0))
    one_hot = torch.zeros(len(list_scalars), one_hot_dim)
    one_hot.scatter_(1, vals, 1.0)
    return one_hot

def _bin_features(list_scalars, feature_dim=2):
    arr = np.array(list_scalars)
    min_val, max_val = np.min(arr), np.max(arr)
    bins = np.linspace(min_val, max_val, num=feature_dim)
    feat = np.digitize(arr, bins) - 1
    assert np.min(feat) == 0
    assert np.max(feat) == feature_dim - 1
    return _one_hot_tensor(feat, one_hot_dim=feature_dim)

# --- Functions from the original git (NOT USED)

def degree_fun(graph, feature_dim):
    graph.node_degree = _one_hot_tensor(
        [d for _, d in graph.G.degree()],
        one_hot_dim=feature_dim)
    return graph

def centrality_fun(graph, **_kwargs):
    nodes = list(graph.G.nodes)
    centrality = nx.betweenness_centrality(graph.G)
    graph.betweenness_centrality = torch.tensor(
        [centrality[x] for x in
        nodes]).unsqueeze(1)
    return graph

def path_len_fun(graph, feature_dim):
    nodes = list(graph.G.nodes)
    graph.path_len = _one_hot_tensor(
        [np.mean(list(nx.shortest_path_length(graph.G,
            source=x).values())) for x in nodes],
        one_hot_dim=feature_dim)
    return graph

def pagerank_fun(graph, **_kwargs):
    nodes = list(graph.G.nodes)
    pagerank = nx.pagerank(graph.G)
    graph.pagerank = torch.tensor([pagerank[x] for x in
        nodes]).unsqueeze(1)
    return graph

def identity_fun(graph, feature_dim):
    graph.identity = compute_identity(graph.edge_index, graph.num_nodes, feature_dim)
    return graph

def clustering_coefficient_fun(graph, feature_dim):
    node_cc = list(nx.clustering(graph.G).values())
    if feature_dim == 1:
        graph.node_clustering_coefficient = torch.tensor(node_cc, dtype=torch.float).unsqueeze(1)
    else:
        graph.node_clustering_coefficient = _bin_features(
                node_cc, feature_dim=feature_dim)

def compute_identity(edge_index, n, k):
    """_summary_

    Parameters
    ----------
    edge_index : LongTensor
        List of edges indices' 
    n : int
        number of nodes
    k : int
        _description_

    Returns
    -------
    _type_
        _description_
    """
    edge_weight = torch.ones((edge_index.size(1),), dtype=torch.float,
                             device=edge_index.device)
    edge_index, edge_weight = pyg_utils.add_remaining_self_loops(edge_index, edge_weight, 1, n)
    adj_sparse = torch.FloatTensor(edge_index, edge_weight,torch.Size([n, n])).to_sparse()
    adj = adj_sparse.to_dense()

    deg = torch.diag(torch.sum(adj, -1))
    deg_inv_sqrt = deg.pow(-0.5)
    adj = deg_inv_sqrt @ adj @ deg_inv_sqrt

    diag_all = [torch.diag(adj)]
    adj_power = adj
    for _ in range(1, k):
        adj_power = adj_power @ adj
        diag_all.append(torch.diag(adj_power))
    diag_all = torch.stack(diag_all, dim=1)
    return diag_all

# In the original git, there is a possibility to preprocess a dataset that is not used here bc it does not correspond to the NM paper.

# --- Added (and used) functions ---
def predict_pretrain(predict):
    """Pretraining function for a NeuroMatch prediction object, 
    it makes the prediction models able to predict if two (embedding like) values are equal or not. 

    Parameters
    ----------
    predict : NM.NeuroMatchPred
    """
    print("Pretrain Prediction Model")
    labels = torch.tensor([1]*32 + [0]*32).to(get_device())
    predict.model.train()
    for _ in range(10000) :
        pos = torch.rand(32,64).to(get_device())
        neg_tar = torch.rand(32,64).to(get_device())
        neg_que = torch.rand(32,64).to(get_device())
        shuf = torch.randperm(64)
        emb_targets = torch.cat((pos, neg_tar), dim=0)[shuf]
        emb_querys = torch.cat((pos, neg_que), dim=0)[shuf]
        predict.train(emb_targets,emb_querys,labels[shuf])
