"""This file contains generator (here meaning *python generator*) 
that infinitely yield examples of graphs issued of real dataset.
"""
#from abc import ABC, abstractmethod
from copy import deepcopy
from torch_geometric.datasets import TUDataset
from torch_geometric.utils.convert import to_networkx

import numpy as np
import networkx as nx


from utils.graph import Generator, k_nodes_walk

# The training process over real graph is not at all implemented this way in the original git.
class RealGenerator(Generator) :
    """Python Generator that infinitely yield examples from a Real Datasets.
    The implemented dataset are the TUDatasets.
     
    This generator aims to mimics the process of the random generator so to use them for the training of the models.
    It yields either a graph of the requested real dataset or a subgraph (sampled from one graph of the dataset)"""
    dataset_str:str = ""
    def __init__(self,dataset_str:str) :
        #Defining new properties of the opbject
        self.dataset_str = dataset_str
        unfilt_ds = TUDataset(root=f"../Testds/{self.dataset_str}", name=self.dataset_str)
        self.ds = []
        for g in unfilt_ds:
            if g.num_nodes < 6 or not nx.is_connected(to_networkx(g,to_undirected=True)) :
                continue
            self.ds.append(g)

        #_sizes_dict is a dictionarry such that {number_of_nodes : [index1,index2,...]}
        # where indexI are the indexs of graph with number_of_nodes in the dataset
        self._sizes_dict = {}
        for i,g in enumerate(self.ds):
            try :
                self._sizes_dict[g.num_nodes].append(i)
            except KeyError : # The KeyError will automatically be raised everytime the algo encountered a new size.
                self._sizes_dict[g.num_nodes] = [i]


        self._saved_sizes_dict = deepcopy(self._sizes_dict)
        self._saved_length = len(self)

        #Defining the scope of graph size.
        self._min_size,self._max_size = min(self._sizes_dict.keys()),max(self._sizes_dict.keys())
        super().__init__(range(self._min_size,self._max_size))
        # Here we are defining possible sizes of graphs (_sizes_list) that can be used (particularly) during the training


    # --- Private Method ---

    def __len__(self):
        length = 0
        for size in self._sizes_dict:
            length += len(self._sizes_dict[size])
        return length

    def _remove(self, size, index) :
        self._sizes_dict[size].remove(index)
        if self._sizes_dict[size] == [] :
            del self._sizes_dict[size] #Removing the key

    # --- Public Method ---
    def generate(self,size=None)-> nx.Graph:
        """_summary_

        Parameters
        ----------
        size : int|None, optional
            Request a size for the generated graphs, by default None

        Returns
        -------
        Networkx.Graph
            returns one of the graph of the dataset OR one subgraph of the dataset (of requested size)
        """
        num_nodes = self._get_size(size)
        if num_nodes in self._sizes_dict.keys() :
            rand_index = np.random.choice(self._sizes_dict[num_nodes], size=1, replace=True)[0]
            nx_graph = to_networkx(self.ds[rand_index],to_undirected=True)
            #-- Removing the used graph from usable graphs
            self._remove(num_nodes,rand_index)
            if len(self) < 0.1 * self._saved_length : #In case too many "little" graph have been erased
                self._sizes_dict = deepcopy(self._saved_sizes_dict) #Reseting usable graphs
            return nx_graph

        else : # Case where no graph have the given size
            possible_sizes = [s for s in self._sizes_dict if s >= num_nodes]
            if len(possible_sizes) == 0 : #If there is no more usable graph larger or the same size as needed size
                self._sizes_dict = deepcopy(self._saved_sizes_dict) #Reseting usable graphs
                return self.generate(size) # To avoid any weird behaviour, it is recursively calling the function to start from the beginning

            sample_graph_size = np.random.choice(possible_sizes, size=1, replace=True)[0]
            rand_index = np.random.choice(self._sizes_dict[sample_graph_size], size=1, replace=True)[0]
            graph = self.ds[rand_index] # Choosing a random larger graph
            start_node = np.random.choice(range(sample_graph_size), size=1, replace=True)[0]
            return k_nodes_walk(graph,start_node,k=num_nodes) #Generating a subgraph of needed size

if __name__ == "__main__":
    gen = RealGenerator("ENZYMES")

    for k in range(1000):
        gen.generate()