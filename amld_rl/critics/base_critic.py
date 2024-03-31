from abc import ABCMeta, abstractmethod
import torch.nn as nn
import torch


class BaseCritic(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def update_critic(self, inputs: torch.Tensor, *args):
        raise NotImplementedError



