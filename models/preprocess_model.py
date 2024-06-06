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

    def node_features_base_fun(self, graph, feature_dim):
            for v in graph.G.nodes:
                if "node_feature" not in graph.G.nodes[v]:
                    graph.G.nodes[v]["node_feature"] = torch.ones(feature_dim)
            return graph


    def augment(self, dataset):
        
        dataset = dataset.apply_transform(self.node_features_base_fun,
            feature_dim=1)
        for key, dim in zip(FEATURE_AUGMENT, FEATURE_AUGMENT_DIMS):
            dataset = dataset.apply_transform(self.node_feature_funs[key], 
                feature_dim=dim)
        return dataset