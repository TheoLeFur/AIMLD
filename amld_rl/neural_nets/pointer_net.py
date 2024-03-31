import math
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch import nn as nn

from amld_rl.neural_nets.attention_module import AttentionModule
from amld_rl.neural_nets.graph_embedding import GraphEmbedding


class InvalidSequenceLength(ValueError):
    pass


class PointerNet(nn.Module):
    def __init__(self,
                 embedding_size: int,
                 hidden_dim: int,
                 seq_len: int,
                 n_glimpses: int,
                 tanh_exploration: Optional[int] = None,
                 use_tanh: Optional[bool] = False,
                 attention: Optional[str] = "D",
                 device: Optional[str] = "cpu") -> None:
        """

        Basic Pointer Network architecture.

        Encoder and decoder are represented as two LSTM layers.

        Encoder: reads the embedded input sequence one city at a time. It transforms it into latent
        memory states enc_1, ... , enc_N

        Decoder: at each step i, dec_i uses the pointing mechanism to produce a distribution over the next
        nodes does to visit, using the above attention module. Once the node is selected, it is passed as input
        to the next decoder step. We repeat this until the input sequence is empty.

        @param embedding_size: Embedding dimension
        @param hidden_dim: Hidden layer dimension
        @param seq_len: Sequence length
        @param n_glimpses: Number of glimpses
        @param tanh_exploration: Exploration factor
        @param use_tanh: If true, we incorporate tanh exploration
        @param device: Device on which we run the computation, cpu by default
        """

        super().__init__()

        self.embedding_size: int = embedding_size
        self.hidden_dim = hidden_dim
        self.n_glimpses = n_glimpses
        self.seq_len = seq_len
        self.device = device

        self.embedding = GraphEmbedding(2, embedding_size, device=self.device)
        self.encoder = nn.LSTM(embedding_size, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(embedding_size, hidden_dim, batch_first=True)

        self.pointer = AttentionModule(
            hidden_dim=self.hidden_dim,
            use_tanh=use_tanh,
            C=tanh_exploration,
            device=self.device,
            attention=attention
        )

        self.glimpse = AttentionModule(
            hidden_dim=self.hidden_dim,
            use_tanh=False,
            device=self.device,
            attention=attention
        )

        self.decoder_start_input = nn.Parameter(
            torch.FloatTensor(embedding_size)).to(self.device)
        self.decoder_start_input.data.uniform_(
            -(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))

    def _mask_logits(
            self,
            logits: torch.Tensor,
            mask: torch.Tensor,
            idxs: torch.Tensor
    ):
        """

        Masks the probabilities of futures decision indices, not allowing the model to see into the future.

        @param logits:
        @param mask: Tensor of shape [BATCH_SIZE, SEQ_LEN]
        @param idxs:
        @return:
        """

        batch_size = logits.size(0)
        clone_mask = mask.clone()

        if idxs is not None:
            clone_mask[[i for i in range(batch_size)], idxs.data] = 1
            logits[clone_mask] = -np.Inf
        return logits, clone_mask

    def forward(self, inputs: torch.Tensor) -> Tuple[List, List]:
        """
        Forward Pass through pointer network
        @param inputs: tensor of shape [BATCH_SIZE, 2, SEQ_LEN]
        @return: Probabilities and action indices
        """

        batch_size: int = inputs.size(0)
        seq_len: int = inputs.size(2)

        if seq_len != self.seq_len:
            raise InvalidSequenceLength(
                f"Tensor sequence length : {seq_len} does not match attribute sequence length{self.seq_len}")

        embedded: torch.Tensor = self.embedding(inputs)
        print(embedded.shape)
        encoder_outputs, (hidden, context) = self.encoder(embedded)
        print(encoder_outputs.shape)
        print(hidden.shape)
        print(context.shape)

        prev_probs: List = []
        prev_idxs: List = []

        mask: torch.Tensor = torch.zeros(
            batch_size, seq_len, device=self.device).to(torch.bool)
        mask.requires_grad = False

        idxs = None
        decoder_input = self.decoder_start_input.unsqueeze(
            0).repeat(batch_size, 1)

        for _ in range(seq_len):
            _, (hidden, context) = self.decoder(
                decoder_input.unsqueeze(1), (hidden, context))
            query = hidden.squeeze(0)
            for _ in range(self.n_glimpses):
                ref, logits = self.glimpse(query, encoder_outputs)
                logits, mask = self._mask_logits(logits, mask, idxs)
                query = torch.bmm(ref, torch.nn.functional.softmax(
                    logits, dim=1).unsqueeze(2)).squeeze(2)

            _, logits = self.pointer(query, encoder_outputs)
            logits, mask = self._mask_logits(logits, mask, idxs)
            probs: torch.Tensor = torch.nn.functional.softmax(
                logits, dim=1)

            idxs = probs.multinomial(1).squeeze(1)
            for old_idxs in prev_idxs:
                if old_idxs.eq(idxs).data.any():
                    idxs = probs.multinomial(1).squeeze(1)
                    break

            decoder_input = embedded[[
                                         i for i in range(batch_size)], idxs.data, :]
            prev_probs.append(probs)
            prev_idxs.append(idxs)

        return prev_probs, prev_idxs


if __name__ == '__main__':
    batch_size = 32
    embedding_dim = 128
    hidden_dim = 128
    n_glimpses = 1

    pointer = PointerNet(
        embedding_size=embedding_dim,
        hidden_dim=hidden_dim,
        n_glimpses=n_glimpses,
        seq_len=16
    )

    input = torch.randn((32, 2, 16))
    pointer(input)
