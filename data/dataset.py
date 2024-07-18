from typing import Callable, Any, List
from torch_geometric.data import InMemoryDataset
import torch
import numpy as np
import data.random_graph_generator as rgg
from utils.torch_ml import to_pyg_data

class SyntheticDataset(InMemoryDataset):
    """Ownmade dataset made of synthetic graph from the random_graph_generator module."""
    def __init__(self, root: str | None = None, transform: Callable[..., Any] | None = None, pre_transform: Callable[..., Any] | None = None,
                pre_filter: Callable[..., Any] | None = None, log: bool = True, force_reload: bool = False,
                graph_sizes:List[int] =  np.arange(20,90)) -> None:
        self.graph_size = graph_sizes
        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) :
        return ['data.raw']

    @property
    def processed_file_names(self) :
        return ['data.pt']

    def download(self) -> None:
        print("Downloading...")
        data_list = []
        gen = rgg.random_generator(self.graph_size)
        for _ in range(1000):
            data_list.append(to_pyg_data(gen.generate()))
        print("Done")
        torch.save(data_list,self.raw_paths[0])

    def process(self) -> None:
        data_list = torch.load(self.raw_paths[0])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

