"""module to generate loaders object that yield OTF Batch of graphs.
This module use Generator from `random_graph_generator` and hence takes networkX graphs as input.
However the OTF_container change types to torch_geometric.data.Data"""
from typing import Tuple, List
from torch_geometric.loader import DataLoader
from utils.torch_ml import to_pyg_data
import data.random_graph_generator as rgg

def gen_data_loaders( mini_epoch:int=100, batch_size:int=64, generator:rgg.Generator=rgg.RandomGenerator) -> Tuple[DataLoader,DataLoader,List[None]]:
    """Generate two On The Fly dataloaders for positive and negative examples for the whole epoch. 
    Each dataloaders wild yield a batch half the sized of total `batch_size`. 

    Parameters
    ----------
    mini_epoch : `int`
        epoch size, default 100
    batch_size : `int`
        size of batch, default 64
    generator : `random_graph_generator.Generator`
        The generator use to create the batchs, default RandomGenerator

    Returns
    -------
    [`Pytorch.Dataloader`, `Pytorch.Dataloader`, [None]]
        Return a list containing one data loader for positive and one for negative examples and a list of None.
        The DataLoader are On-The-Fly Generator
    """
    loaders = []
    single_loader_batch_size = batch_size//2
    for _ in range(2):
        dataset_length = single_loader_batch_size * mini_epoch
        OTF_ds = OTFContainer(generator,dataset_length)

        loaders.append(DataLoader(OTF_ds, batch_size=single_loader_batch_size))
    loaders.append([None]*(mini_epoch))
    return loaders


class OTFContainer(list) :
    """On-The-Fly Container that iterate over a graph generator object"""
    def __init__(self,generator:rgg.Generator,length:int) :
        """The OTFContainer generate a sample of an object using `generator` when it is needed. 
        The Container have a given `length`

        Parameters
        ----------
        generator : rgg.Generator
            Generates a networkX graph
        length : int
            length of the container
        """
        super().__init__()
        self._generator = generator
        self._len = length
        self._index = 0

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        for _ in range(self._len):
            yield to_pyg_data(self._generator.generate())

    def __next__(self) :
        if self._index < self._len :
            self._index += 1
            return to_pyg_data(self._generator.generate())

    def __getitem__(self,i):
        if isinstance(i,slice) :
            OTF_L = []
            for _ in range(self._len)[i]:
                OTF_L.append(to_pyg_data(self._generator.generate()))
            return OTF_L
        else :
            return to_pyg_data(self._generator.generate())
