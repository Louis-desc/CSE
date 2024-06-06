"""Generating Batch"""
import random
import torch
import networkx as nx
from torch_geometric import data as pyg_data
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils.convert import to_networkx


from utils.torch_ml import to_pyg_data
import data.random_graph_generator as rgg

def augment_batch(pos_target:pyg_data.Batch, neg_target:pyg_data.Batch, generator:rgg.Generator) :   #Refering to gen_batch functions in original git
    """Generate a batch of data with positive and negative examples. 
    Samples subgraphs from two target graphs (one for positive and one for negative examples). 

    Parameters
    ----------
    pos_target : torch_geometric.data.Batch
        Target graph for positive examples. 
        Batch(G=[epoch_size], batch=[183], edge_index=[2, 1348], edge_label_index=[2, 1348], node_label_index=[183])
    neg_target : torch_geometric.data.Batch
        Target graph for negative examples
    batch_neg_query : torch_geometric.data.Batch
        Randomly generated query graph 
    generator : random_graph_generator.Generator
        generator object to generate graphs. 

    Returns
    -------
    _type_
        _description_
    """

    # ---- Positive examples ----
    target = []
    query = []
    for graph in pos_target.to_data_list():
        g_tar, g_quer = sample_subgraph(graph,True)
        target.append(g_tar)
        query.append(g_quer)
    pos_query = pyg_data.Batch.from_data_list(query)
    pos_target = pyg_data.Batch.from_data_list(target)
    # ------    ------

    #batch_neg_query = [to_pyg_data(generator.generate(size=len(g))) if i not in hard_neg_idxs
    #                   else g for i, g in enumerate(neg_target.to_data_list())]
    #The question is why do they generate it here better than in the Dataloader

    # ---- Negative examples ----
    hard_neg_idxs = set(random.sample(range(neg_target.batch_size),
        int(neg_target.batch_size * 1/2)))

    query = []
    target = []
    for i,g in enumerate(neg_target.to_data_list()) :
        if i not in hard_neg_idxs :                     #For easy negatives
            g_quer = to_pyg_data(generator.generate(size=len(g)))

            fake_anchor = random.choice(list(range(g_quer.num_nodes)))
            g_quer.x = gen_anchor_feature(g_quer,fake_anchor) #Adding a fake anchor_feature for the query

            g_tar = g #Normally, g is already a deep_copy of our initial batch 
            fake_anchor = random.choice(list(range(g_tar.num_nodes)))
            g_tar.x = gen_anchor_feature(g_tar,fake_anchor) #Adding a fake anchor_feature for the target

        else:                                           #For hard negatives
            g_tar, g_quer = sample_subgraph(g,train=True,use_hard_neg=True)
        query.append(g_quer)
        target.append(g_tar)
    neg_query = pyg_data.Batch.from_data_list(query)
    neg_target = pyg_data.Batch.from_data_list(target)
    # ------    ------

    # ---- Adding more features ----
    # pos_target = augmenter.augment(pos_target).to(get_device())
    # pos_query = augmenter.augment(pos_query).to(get_device())
    # neg_target = augmenter.augment(neg_target).to(get_device())
    # neg_query = augmenter.augment(neg_query).to(get_device())
    #
    # I delete this part of the code bc when using the original git without modification
    # you only use this augment to be sure every batch have at least 1 feature for the model training.
    # It is always the case with this algorithm.
    return pos_target, pos_query, neg_target, neg_query


def sample_subgraph(graph:pyg_data.Data, train:bool, use_hard_neg:bool=False) :
    """Given a target `graph` sample a query that match (or don't match if `use_hard_neg` = True). 
    The query is anchored to a node. This node is indicated by a feature (one hot vector). 
    
    Parameters
    ----------
    graph : pyg_data.Data
        The initial target graph to sample a query from. 
    train : bool
        Indicate if the function is use in the training process or not. 
        If it is, k (from k-hop-neighborhood) can't be equal to the number of nodes in the graph.
    use_hard_neg : bool, optional
        Indicate is the given graph is part of hard neg batch, by default False.
        If equals to true, the query subgraph sampled is modified.

    Returns
    -------
    target_graph : pyg_data.Data, query_graph : pyg_data.Data
        Returns two pyg graph. The first one is the initial target graph (with anchor_feature) 
        and the second one is the sampled query (with anchor_feature). 
    """
    min_size = 5
    d = 1 if train else 0                                #I'm not sure of why they use that
    k = random.randint(min_size - d,graph.num_nodes-1)   #Choose the k for the k-neighborhood
    start_node = random.choice(list(range(graph.num_nodes)))#Choose the starting point of the walk
    graph.x = gen_anchor_feature(graph,start_node)

    # ---- Query graph ----
    neighood = k_hop_subgraph(start_node,k,graph.edge_index)
    ng_graph = pyg_data.Data(edge_index=neighood[1],num_nodes=int(len(neighood[0])))
    ng_graph.x = gen_anchor_feature(ng_graph,neighood[2])
    # ------    ------
    #For more details, check k_hop_subgraph doc (it doesn't actually return a pyG.Data object)

    # ---- Negative examples handeling ----
    if use_hard_neg and train :
        nx_ng = to_networkx(ng_graph,to_undirected=True)
        saved_x = ng_graph.x
        non_edges = list(nx.non_edges(nx_ng))
        for u, v in random.sample(non_edges, random.randint(1, min(max(len(non_edges),2),5))): #Adding up to a maximum of 5 edges more
            nx_ng.add_edge(u,v)
        ng_graph = to_pyg_data(nx_ng)
        ng_graph.x = saved_x                                                             #Here I use exactly the same anchor. It could be modified.


    return graph, ng_graph



def gen_anchor_feature(graph:pyg_data.Data,anchor:int)->torch.Tensor:
    """Generate a one-hot vector for anchor_feature

    Parameters
    ----------
    graph : pyg_data.Data
        the graph to feat
    anchor : int
        the anchor node idx

    Returns
    -------
    torch.Tensor
        torch Tensor of the feature
    """
    anchor_feature =[]
    for k in range(graph.num_nodes):
        if k == anchor :
            anchor_feature.append(torch.ones(1))
        else :
            anchor_feature.append(torch.zeros(1))
    return torch.tensor(anchor_feature)
