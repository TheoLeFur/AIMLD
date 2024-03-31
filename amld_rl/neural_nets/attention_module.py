import math
from typing import Optional

import torch
from torch import nn as nn


class AttentionModule(nn.Module):

    def __init__(
            self,
            hidden_dim: int,
            use_tanh: Optional[bool] = False,
            C: Optional[int] = 10,
            device: Optional[str] = "cpu",
            attention: Optional[str] = "D"
    ) -> None:
        """

        Instance of Bahdanau attention. It takes an input a query tensor = dec_i and a set of reference
        vectors {enc_1, ..., enc_} and predicts a probability distribution over the references.
        This probability distribution represents the degree to which the model points to reference enc_i upon
        seeing the query tensor q.

        We have two attention matrices, represented as two linear layers Wr and Wq.
        Our attention function computes:

        u_i = <v, tanh(Wr@r_i, Wq@q)>

        Optionally, we apply C * tanh to the above

        @param hidden_dim: Hidden dim size
        @param use_tanh: If true, logits are passed through tanh activation
        @param C: Hyperparameter controlling the range of the logits, and therefore the entropy of the distribution induces
        by the attention function
        @param device: Device name
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_tanh = use_tanh
        self.tanh = nn.Tanh()
        self.C = C

        self.device = device
        self.attention = attention

        # Query and ref projections
        self.Wq = nn.Linear(hidden_dim, hidden_dim)
        self.Wr = nn.Conv1d(hidden_dim, hidden_dim, 1, 1)
        value: torch.Tensor = torch.FloatTensor(self.hidden_dim)
        self.value = nn.Parameter(value).to(self.device)

        # Weights initialization
        self.value.data.uniform_(-(1. / math.sqrt(hidden_dim)),
                                 1. / math.sqrt(hidden_dim))

    def forward(self, query: torch.Tensor, ref: torch.Tensor):
        """
        @param query: query vector, corresponding to the ith output of the decoder, i = 1, ... , seq_len
        @param ref: reference vector, corresponding to enc_1,...,enc_k
        @return: The reference vectors and the logits
        """

        batch_size: int = ref.size(0)
        seq_len: int = ref.size(1)

        if self.attention == "BHD":
            ref = ref.permute(0, 2, 1)
            query: torch.Tensor = self.Wq(query).unsqueeze(2)
            ref: torch.Tensor = self.Wr(ref)

            query = query.repeat(1, 1, seq_len)
            value: torch.Tensor = self.value.unsqueeze(
                0).unsqueeze(0).repeat(batch_size, 1, 1)

            logits: torch.Tensor = torch.bmm(
                value, self.tanh(query + ref)).squeeze(1)
        elif self.attention == "D":
            query = query.unsqueeze(2)
            logits = torch.bmm(ref, query).squeeze(2)
            ref = ref.permute(0, 2, 1)
        else:
            raise NotImplementedError

        if self.use_tanh:
            # Optional exploration
            logits = self.C * self.tanh(logits)

        return ref, logits
