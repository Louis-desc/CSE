import random
import numpy as np
import torch
import torch.utils.data as tud

from deepsnap.graph import Graph as DSGraph
from deepsnap.batch import Batch
from deepsnap import dataset

from data.random_graph_generator import ERGenerator
from models.preprocess_model import FeatureAugment
from utils.torch_ml import get_device


def gen_data_loaders( epoch_size:int=1000, batch_size:int=64, generator:dataset.Generator=ERGenerator, use_distributed_sampling=False):
    """generate a list that contains two batch for positive and negative example

    Parameters
    ----------
    epoch_size : int
        epoch size, default 1000
    batch_size : int
        size of batch, default 64
    generator : generator object from deepSnap
        The generator use to create the batchs, default ERGenerator
    use_distributed_sampling : bool, optional
        Not used, default False

    Returns
    -------
    [Pytorch.Dataloader, Pytorch.Dataloader, [None]]
        Return a list containing one data loader for positive and one for negative examples and a list of None
    """
    loaders = []
    for _ in range(2):
        graphs_ds = dataset.GraphDataset( None, task="graph", generator=generator)

        #dataset = get_dataset("graph", size // 2, np.arange(self.min_size + 1, self.max_size + 1))
        #sampler = tud.distributed.DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank()) if use_distributed_sampling else None

        loaders.append(tud.DataLoader(graphs_ds,collate_fn=Batch.collate([]),
                                      batch_size=batch_size // 2, sampler=None, shuffle=False))
    loaders.append([None]*(epoch_size))
    return loaders

def gen_batch(pos_target, neg_target, batch_neg_query, train, generator) :
    """Generate a batch of data with positive and negative examples. 
    Samples subgraphs from two target graphs (one for positive and one for negative examples). 

    Parameters
    ----------
    pos_target : _type_
        Target graph for positive examples
    neg_target : _type_
        Target graph for negative examples
    batch_neg_query : _type_
        _description_
    train : _type_
        _description_
    generator : _type_
        generator object to generate graphs. 

    Returns
    -------
    _type_
        _description_
    """
    augmenter = FeatureAugment()

    pos_target, pos_query = pos_target.apply_transform_multi(sample_subgraph)

    # TODO: use hard negs
    hard_neg_idxs = set(random.sample(range(len(neg_target.G)),
        int(len(neg_target.G) * 1/2)))
    #hard_neg_idxs = set()e

    batch_neg_query = Batch.from_data_list(
        [DSGraph(generator.generate(size=len(g)) if i not in hard_neg_idxs else g)
        for i, g in enumerate(neg_target.G)]
        )
    
    for i, g in enumerate(batch_neg_query.G):
        g.graph["idx"] = i
    
    _, neg_query = batch_neg_query.apply_transform_multi(sample_subgraph,
        hard_neg_idxs=hard_neg_idxs)

    # Anchored
    neg_target = neg_target.apply_transform(add_anchor)
    pos_target = augmenter.augment(pos_target).to(get_device())
    pos_query = augmenter.augment(pos_query).to(get_device())
    neg_target = augmenter.augment(neg_target).to(get_device())
    neg_query = augmenter.augment(neg_query).to(get_device())
    #print(len(pos_target.G[0]), len(pos_query.G[0]))
    return pos_target, pos_query, neg_target, neg_query



def sample_subgraph(graph, offset=0, use_precomp_sizes=False, filter_negs=False, supersample_small_graphs=False, neg_target=None, hard_neg_idxs=None):
    if neg_target is not None:
        graph_idx = graph.G.graph["idx"]
    use_hard_neg = (hard_neg_idxs is not None and graph.G.graph["idx"] in hard_neg_idxs)
    done = False
    n_tries = 0
    while not done:
        if use_precomp_sizes:
            size = graph.G.graph["subgraph_size"]
        else:
            if train and supersample_small_graphs:
                sizes = np.arange(self.min_size + offset,
                    len(graph.G) + offset)
                ps = (sizes - self.min_size + 2) ** (-1.1)
                ps /= ps.sum()
                size = stats.rv_discrete(values=(sizes, ps)).rvs()
            else:
                d = 1 if train else 0
                size = random.randint(self.min_size + offset - d,
                    len(graph.G) - 1 + offset)
        start_node = random.choice(list(graph.G.nodes))
        neigh = [start_node]
        frontier = list(set(graph.G.neighbors(start_node)) - set(neigh))
        visited = set([start_node])
        while len(neigh) < size:
            new_node = random.choice(list(frontier))
            assert new_node not in neigh
            neigh.append(new_node)
            visited.add(new_node)
            frontier += list(graph.G.neighbors(new_node))
            frontier = [x for x in frontier if x not in visited]
        if self.node_anchored:
            anchor = neigh[0]
            for v in graph.G.nodes:
                graph.G.nodes[v]["node_feature"] = (torch.ones(1) if
                    anchor == v else torch.zeros(1))
                #print(v, graph.G.nodes[v]["node_feature"])
        neigh = graph.G.subgraph(neigh)
        if use_hard_neg and train:
            neigh = neigh.copy()
            if random.random() < 1.0 or not self.node_anchored: # add edges
                non_edges = list(nx.non_edges(neigh))
                if len(non_edges) > 0:
                    for u, v in random.sample(non_edges, random.randint(1,
                        min(len(non_edges), 5))):
                        neigh.add_edge(u, v)
            else:                         # perturb anchor
                anchor = random.choice(list(neigh.nodes))
                for v in neigh.nodes:
                    neigh.nodes[v]["node_feature"] = (torch.ones(1) if
                        anchor == v else torch.zeros(1))

        if (filter_negs and train and len(neigh) <= 6 and neg_target is
            not None):
            matcher = nx.algorithms.isomorphism.GraphMatcher(
                neg_target[graph_idx], neigh)
            if not matcher.subgraph_is_isomorphic(): done = True
        else:
            done = True

    return graph, DSGraph(neigh)


def add_anchor(g, anchors=None):
    if anchors is not None:
        anchor = anchors[g.G.graph["idx"]]
    else:
        anchor = random.choice(list(g.G.nodes))
    for v in g.G.nodes:
        if "node_feature" not in g.G.nodes[v]:
            g.G.nodes[v]["node_feature"] = (torch.ones(1) if anchor == v
                else torch.zeros(1))
    return g
