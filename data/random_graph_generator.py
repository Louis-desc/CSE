"""Classes for random graph generators directly from Neuromatch

In this module only networkX graphs are generated.
In this module : 
- Erdos Renyi Generator : ERGenerator  
- Watt-Strogatz Generator : WSGenerator
- Barabasi-Albert Generator : BAGenerator
- Power Law Cluster Generator : PLCGenerator
"""

from abc import ABC, abstractmethod
import random
import networkx as nx
import numpy as np

# ----  Abstract class  ----

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

# ----  Generators implementation  ----

class ERGenerator(Generator):
    """A modified Erdos-Renyi model generator
    """
    def __init__(self, sizes, alpha=1.3, **kwargs):
        """A modified implementation of Erdos-Renyi model generator
        
        This implementation use a beta distribution of parameter alpha and
        a mean value of log(size)/size to choose randomly the usual p parameter. 
        
        > p is the probability for each possible edge to exist. 
        
        This implementation only returns connected graphs. 

        Parameters
        ----------
        sizes : [int]
            A list of different possible sizes (numbers of nodes) for the generated graphs
        alpha : float, optional
            alpha parameter for the beta distribution that generate p, by default 1.3
        """
        super().__init__(sizes, **kwargs)
        self.alpha = alpha

    def generate(self, size=None):
        num_nodes = self._get_size(size)
        alpha = self.alpha
        mean = np.log2(num_nodes) / num_nodes

        # p follows beta distribution with mean = log2(num_graphs) / num_graphs
        beta = alpha / mean - alpha
        p = np.random.beta(alpha, beta)

        graph = nx.gnp_random_graph(num_nodes, p)

        while not nx.is_connected(graph):
            #print(f"ER : num_nodes : {num_nodes}, p  : {p}")
            p = np.random.beta(alpha, beta)
            graph = nx.gnp_random_graph(num_nodes, p)
        return graph

class WSGenerator(Generator):
    """A modified Watt-Strogatz model generator"""
    def __init__(self, sizes, k_alpha=1.3,
            rewire_alpha=2., rewire_beta=2., **kwargs):
        """A modified Watt-Strogatz model generator
        
        This implementation use beta distributions to generate : 
        - The average degree $k$ used in the original algorithm. Here $k$ >= 2. 
        - The rewiring probability used in the original algorithm
        
        This implementation only returns connected graphs.

        Parameters
        ----------
        sizes : [int]
            A list of different possible sizes (numbers of nodes) for the generated graphs
        k_alpha : float, optional
            The alpha parameter for the beta distribution for k, by default 1.3
        rewire_alpha : float, optional
            The alpha parameter for the beta distribution for rewire probability, by default 2.
        rewire_beta : float, optional
            The beta parameter for the beta distribution for rewire probability, by default 2.
        """
        super().__init__(sizes, **kwargs)
        self.k_alpha = k_alpha
        self.rewire_alpha = rewire_alpha
        self.rewire_beta = rewire_beta


    def generate(self, size=None):
        num_nodes = self._get_size(size)

        # Computing k
        k_alpha = self.k_alpha
        k_mean = np.log2(num_nodes) / num_nodes
        k_beta = k_alpha / k_mean - k_alpha
        k = int(np.random.beta(k_alpha, k_beta) * num_nodes)
        k = max(k, 2)

        # Computing rewiring probability
        rewire_alpha = self.rewire_alpha
        rewire_beta = self.rewire_beta
        p = np.random.beta(rewire_alpha, rewire_beta)

        #Generating graph
        graph = nx.watts_strogatz_graph(num_nodes,k,p)
        while not nx.is_connected(graph) :
            try :
                #print(f"WS : num_nodes : {num_nodes}, k  : {k}, p : {p}")
                graph = nx.connected_watts_strogatz_graph(num_nodes, k, p)
            finally :
                pass
        return graph

class BAGenerator(Generator):
    """A modified Barabasi Albert model generator"""
    def __init__(self, sizes, **kwargs):
        """A modified Barabasi Albert model generator
        
        The usual $m$ parameter of the BA model is here 2 * np.log2(size)
        
        This implementation also use the networkX extended model that, for each step choose between :
        - adding a new edge to the existing graph with probability $p$ (here p <= 0.2)
        - rewiring existing edges with a probability $q$ (here q <= 0.2)
        - or adding $m$ new nodes to the graph and creating edges as in normal BA model. 
        
        This implementation only returns connected graphs

        Parameters
        ----------
        sizes : [int]
            A list of different possible sizes (numbers of nodes) for the generated graphs
        """
        super().__init__(sizes, **kwargs)
        self.max_p = 0.2
        self.max_q = 0.2

    def generate(self, size=None):
        num_nodes = self._get_size(size)
        max_m = int(2 * np.log2(num_nodes))
        # m = np.random.choice(max_m) + 1 in the original git
        m = random.randint(1,max_m)
        if m== num_nodes :
            m-= 1
        p = np.min([np.random.exponential(20), self.max_p])
        q = np.min([np.random.exponential(20), self.max_q])

        graph = nx.extended_barabasi_albert_graph(num_nodes, m, p, q)
        while not nx.is_connected(graph):
            #print(f"BA : num_nodes : {num_nodes}, m  : {m}, p : {p}, q : {q}")
            graph = nx.extended_barabasi_albert_graph(num_nodes, m, p, q)
        return graph

class PowerLawClusterGenerator(Generator):
    """A modified Power Law Cluster model generator"""
    def __init__(self, sizes, max_triangle_prob=0.5, **kwargs):
        """A modified Power Law Cluster model generator
        
        The Power Law clustering generator is already a modified version of the BA generator that 
        adds triangle when creating new edges with probability $p$. 
        
        Here, $p$ is randomly chosen with a uniform distribution between in (0,max_triangle_prob)
        
        This implementation only returns connected graphs. 

        Parameters
        ----------
        sizes : [int]
            A list of different possible sizes (numbers of nodes) for the generated graphs
        max_triangle_prob : float, optional
            maximum triangle probability used for the uniform distribution, by default 0.5
        """
        super().__init__(sizes, **kwargs)
        self.max_triangle_prob = max_triangle_prob

    def generate(self, size=None):
        num_nodes = self._get_size(size)
        max_m = int(2 * np.log2(num_nodes))
        m = np.random.choice(max_m) + 1
        triangle_prob = np.random.uniform(high=self.max_triangle_prob)

        graph = nx.powerlaw_cluster_graph(num_nodes, m, triangle_prob)
        while not nx.is_connected(graph):
            if num_nodes == m :
                #For some reason, nx.powerlaw... don't actually works and always send back a set of nodes without any edges
                m -= 1
            #print(f"PLC : num_nodes : {num_nodes}, m  : {m}, triangle_prob : {triangle_prob}")
            graph = nx.powerlaw_cluster_graph(num_nodes, m, triangle_prob)

        return graph
#Alias
PLCGenerator = PowerLawClusterGenerator

class RandomGenerator(Generator) :
    """Generator object that randomly choose a generator from a graph-generator list.
    """
    def __init__(self,sizes,gen_list=None) :
        super().__init__(sizes)
        self.gen_list = gen_list
        self.gen_prob = np.ones(len(self.gen_list)) / len(self.gen_list)

    @property
    def gen_list(self):
        """Returns generator list property of the object.
        """
        return self._gen_list

    @gen_list.setter
    def gen_list(self,val):
        if val is None :
            sizes = self._sizes_list
            self._gen_list = [ERGenerator(sizes),WSGenerator(sizes),BAGenerator(sizes),PowerLawClusterGenerator(sizes)]
        
        else:
            assert hasattr(val, '__iter__') #Check if the value is a list / An object you can iterate over
            self._gen_list = []
            for gen in val :
                if isinstance(gen,Generator): #Check if the value is a generator
                    self._gen_list.append(gen)
                else:
                    assert issubclass(gen,Generator) #Check if the value is a generator
                    self._gen_list.append(gen(sizes))

    def generate(self, size=None):
        """_summary_

        Parameters
        ----------
        size : `int`|None, optional
            If `int` type : set the generated graph size. If None choose randomly in sizes, by default None

        Returns
        -------
        `torch_geometric.data.Data`
            A graph object from networkx
        """
        gen = np.random.choice(self.gen_list, p=self.gen_prob)
        return gen.generate(size = size)

def random_generator(sizes) :
    """Create a generator that, each time it is called, ramdomly sample a graph from one of the following
    generators : 
    - Erdos Renyi Generator : ERGenerator(alpha=1.3) 
    - Watt-Strogatz Generator : WSGenerator(k_alpha=1.3, rewire_alpha=2., rewire_beta=2.)
    - Barabasi-Albert Generator : BAGenerator()
    - Power Law Cluster Generator : PLCGenerator(triangle_prob=0.5)
    (With all their default value from this package)

    Parameters
    ----------
    sizes : [int]
        A list of different possible sizes (numbers of nodes) for the generated graphs

    Returns
    -------
    deepsnap.dataset.EnsembleGenerator
        list of generators that generate graphs randomly (and uniformely) from one of the generator of the list
    """
    generator = RandomGenerator(sizes,
            [ERGenerator(sizes),
            WSGenerator(sizes),
            BAGenerator(sizes),
            PowerLawClusterGenerator(sizes)])
    return generator
