from torch.utils.data import Dataset
from typing import Optional
import torch

class BaseDatasetGenerator(Dataset):

    def __init__(self, random_seed : Optional[int] = None):
        super().__init__()

        if random_seed is not None:
            torch.manual_seed(random_seed)


