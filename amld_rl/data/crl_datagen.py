from torch.utils.data import Dataset
from typing import Optional
from tqdm import tqdm
import torch
from amld_rl.data.base_datagen import BaseDatasetGenerator


class TSPDatasetGenerator(BaseDatasetGenerator):

    def __init__(
            self,
            num_nodes: int,
            num_samples: int,
            random_seed: Optional[int] = None
    ) -> None:
        """
        Dataset generator for the Travelling Salesman Problem. We assume that the value
        of the nodes in the graph are normalized to fit in [0,1].

        @param num_nodes: Number of nodes in the graph
        @param num_samples: Number of samples in the dataset
        @param random_seed: Optional seed
        """
        super().__init__(random_seed)

        self.data_set = []
        for _ in tqdm(range(num_samples)):
            x = torch.FloatTensor(2, num_nodes).uniform_(0, 1)
            self.data_set.append(x)

        self.size = len(self.data_set)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data_set[idx]
