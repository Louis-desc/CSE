"""package to generate loaders object that yield OTF Batch of graphs."""
from torch_geometric.utils.convert import from_networkx
from torch_geometric.loader import DataLoader
import data.random_graph_generator as rgg

def gen_data_loaders( epoch_size:int=1000, batch_size:int=64, generator:rgg.Generator=rgg.RandomGenerator):
    """Generate two On The Fly dataloaders for positive and negative examples for the whole epoch. 
    Each dataloaders wild yield a batch half the sized of total `batch_size`. 

    Parameters
    ----------
    epoch_size : `int`
        epoch size, default 1000
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
        dataset_length = single_loader_batch_size * epoch_size
        OTF_ds = OTFContainer(generator,dataset_length)

        loaders.append(DataLoader(OTF_ds, batch_size=single_loader_batch_size))
    loaders.append([None]*(epoch_size))
    return loaders


class OTFContainer(list) :
    """On-The-Fly Container that iterate over a generator object"""
    def __init__(self,generator:rgg.Generator,length:int) :
        """The OTFContainer generate a sample of an object using `generator` when it is needed. 
        The Container have a given `length`

        Parameters
        ----------
        generator : rgg.Generator
            Generates a graph
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
            yield from_networkx(self._generator.generate())

    def __next__(self) :
        if self._index < self._len :
            self._index += 1
            return from_networkx(self._generator.generate())

    def __getitem__(self,i):
        if isinstance(i,slice) :
            OTF_L = []
            for _ in range(self._len)[i]:
                OTF_L.append(from_networkx(self._generator.generate()))
            return OTF_L
        else :
            return from_networkx(self._generator.generate())
