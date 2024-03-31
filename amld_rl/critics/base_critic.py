from abc import ABC
import torch.nn as nn


class BaseCritic(nn.Module):
    def __init__(self):
        super().__init__()
