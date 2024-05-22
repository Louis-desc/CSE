"""Neural model used in the preprocessing to prepare the data to the NM algorithm. 
"""
# --- Imports
import numpy as np
import torch
from torch import nn
import utils.preprocess as up
# import torch_geometric.utils as pyg_utils
# from torch_scatter import scatter_add

# --- Supposed to be constant
AUGMENT_METHOD = "concat"
FEATURE_AUGMENT, FEATURE_AUGMENT_DIMS = [], []

# --- Preprocessings models

class FeatureAugment(nn.Module):
    def __init__(self):
        super().__init__()

        # def motif_counts_fun(graph, feature_dim): 
        #     assert feature_dim % 73 == 0
        #     counts = orca.orbit_counts("node", 5, graph.G)
        #     counts = [[np.log(c) if c > 0 else -1.0 for c in l] for l in counts]
        #     counts = torch.tensor(counts).type(torch.float)
        #     #counts = FeatureAugment._wave_features(counts,
        #     #    feature_dim=feature_dim // 73)
        #     graph.motif_counts = counts
        #     return graph


        self.node_feature_funs = {"node_degree": up.degree_fun,
            "betweenness_centrality": up.centrality_fun,
            "path_len": up.path_len_fun,
            "pagerank": up.pagerank_fun,
            'node_clustering_coefficient': up.clustering_coefficient_fun,
            # "motif_counts": motif_counts_fun,
            "identity": up.identity_fun}

    def register_feature_fun(self, name, feature_fun):
        self.node_feature_funs[name] = feature_fun
        
    def node_features_base_fun(self, graph, feature_dim):
            for v in graph.G.nodes:
                if "node_feature" not in graph.G.nodes[v]:
                    graph.G.nodes[v]["node_feature"] = torch.ones(feature_dim)
            return graph

    @staticmethod
    def _wave_features(list_scalars, feature_dim=4, scale=10000):
        pos = np.array(list_scalars)
        if len(pos.shape) == 1:
            pos = pos[:,np.newaxis]
        batch_size, n_feats = pos.shape
        pos = pos.reshape(-1)

        rng = np.arange(0, feature_dim // 2).astype(
            np.float) / (feature_dim // 2)
        sins = np.sin(pos[:,np.newaxis] / scale**rng[np.newaxis,:])
        coss = np.cos(pos[:,np.newaxis] / scale**rng[np.newaxis,:])
        m = np.concatenate((coss, sins), axis=-1)
        m = m.reshape(batch_size, -1).astype(np.float)
        m = torch.from_numpy(m).type(torch.float)
        return m


    def augment(self, dataset):
        dataset = dataset.apply_transform(self.node_features_base_fun,
            feature_dim=1)
        for key, dim in zip(FEATURE_AUGMENT, FEATURE_AUGMENT_DIMS):
            dataset = dataset.apply_transform(self.node_feature_funs[key], 
                feature_dim=dim)
        return dataset