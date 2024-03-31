import math
from typing import Optional, List

import torch
from torch import nn as nn


class GraphEmbedding(nn.Module):
    def __init__(
            self,
            input_size: int,
            embedding_size: int,
            device: Optional[str] = "cpu"
    ) -> None:
        """
        Graph embedding to encode the nodes of the graph in a latent space
        @param input_size: size of input
        @param embedding_size: size of latent space
        """

        super().__init__()

        self.device: str = device
        self.embedding_size: int = embedding_size
        self.embedding: torch.Tensor = nn.Parameter(
            torch.FloatTensor(input_size, embedding_size)).to(self.device)

        # Weights initialization
        self.embedding.data.uniform_(-(1. / math.sqrt(embedding_size)),
                                     1. / math.sqrt(embedding_size))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through embedding.
        @param inputs:
        @return: embedded input tensor
        """
        # Batch size and sequence length
        batch_size: int = inputs.size(0)
        seq_len: int = inputs.size(2)

        embedding: torch.Tensor = self.embedding.repeat(batch_size, 1, 1)
        embedded: List = []

        inputs: torch.Tensor = inputs.unsqueeze(1).to(self.device)

        for i in range(seq_len):
            embedded.append(torch.bmm(inputs[:, :, :, i].float(), embedding))
        embedded_tensor: torch.Tensor = torch.cat(embedded, 1).to(self.device)
        return embedded_tensor
