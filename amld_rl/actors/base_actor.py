import torch.nn as nn


class BaseActor(nn.Module):

    def __init__(self):
        super().__init__()
