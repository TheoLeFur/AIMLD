import math
import os.path
from typing import Callable, Optional, List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from amld_rl.models.abstract_model import BaseModel


# ERRORS
class InvalidSequenceLength(ValueError):
    pass


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
        self.embedding: nn.Parameter = nn.Parameter(
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

        embedding: nn.Parameter = self.embedding.repeat(batch_size, 1, 1)
        embedded: List = []

        inputs: torch.Tensor = inputs.unsqueeze(1).to(self.device)

        for i in range(seq_len):
            embedded.append(torch.bmm(inputs[:, :, :, i].float(), embedding))
        embedded_tensor: torch.Tensor = torch.cat(embedded, 1).to(self.device)
        return embedded_tensor


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
        @param inputs: tensor of shape [BATCH_SIZE, 1, SEQ_LEN]
        @return: Probabilities and action indices
        """

        batch_size: int = inputs.size(0)
        seq_len: int = inputs.size(2)

        if seq_len != self.seq_len:
            raise InvalidSequenceLength(
                f"Tensor sequence length : {seq_len} does not match attribute sequence length{self.seq_len}")

        embedded: torch.Tensor = self.embedding(inputs)
        encoder_outputs, (hidden, context) = self.encoder(embedded)

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


class CombinatorialRL(nn.Module):
    def __init__(self,
                 embedding_size: int,
                 hidden_dim: int,
                 seq_len: int,
                 n_glimpses: int,
                 reward: Callable,
                 tanh_exploration: Optional[int] = 10,
                 use_tanh: Optional[bool] = False,
                 attention: Optional[str] = "D",
                 device: Optional[str] = "cpu"
                 ):
        """

        @param embedding_size: Dimension of sequence embedding
        @param hidden_dim: Dimensions of hidden layer
        @param seq_len: Input sequence length
        @param n_glimpses: Number of glimpses used in the Pointer Network
        @param reward: Reward function
        @param tanh_exploration: Optional: exploration hyperparameter
        @param use_tanh: If true, use tanh exploration
        @param device: Device name, defaults to "cpu"

        """
        super().__init__()

        self.reward = reward
        self.device = device

        self.actor = PointerNet(
            embedding_size,
            hidden_dim,
            seq_len,
            n_glimpses,
            tanh_exploration,
            use_tanh,
            attention,
            self.device,
        )

    def greedy_sample(
            self,
            input_graph: torch.Tensor,
            n_samples: Optional[int] = 100
    ) -> Tuple[torch.Tensor, List]:
        """

        Samples greedily from optimal policy to find the best solution to the TSP problem,

        @param input_graph: tensor of shape [2, SEQ_LEN]
        @param n_samples: Number of samples
        @return: best reward and best action
        """

        best_reward = np.Inf
        best_action: List = []

        for _ in range(n_samples):
            input_graph = input_graph.to(self.device)
            R, action_probs, actions, action_indices = self(
                input_graph.unsqueeze(0))
            if R < best_reward:
                best_reward = R
                best_action = actions

        return R, best_action

    def save_weights(self, epoch: int, path: str, exp_name: str = "1e5DS") -> None:
        """
        Save the weights of model

        @param epoch: Epoch number
        @param path: Path name
        @return: None
        """
        torch.save(self.state_dict(), os.path.join(
            f"{path}", f"checkpoint-{exp_name}-{epoch}.pth"))

    def load_weights(self, path: str) -> None:
        """
        Load weights from path

        @param path: Path name
        @return: None
        """
        self.load_state_dict(torch.load(path))

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs: [batch_size, input_size, seq_len]
        """
        inputs = inputs.to(self.device)
        batch_size = inputs.size(0)

        probs, action_idxs = self.actor(inputs)

        actions = []
        inputs = inputs.transpose(1, 2)
        for action_id in action_idxs:
            actions.append(inputs[torch.tensor(
                [x for x in range(batch_size)], device=self.device), action_id.data, :])

        action_probs = []
        for prob, action_id in zip(probs, action_idxs):
            action_probs.append(prob[torch.tensor(
                [x for x in range(batch_size)], device=self.device), action_id.data])

        R = self.reward(actions, self.device)

        return R, action_probs, actions, action_idxs


def compute_actor_objective(
        advantage: torch.Tensor,
        probs: List
) -> torch.Tensor:
    log_probs = 0
    for prob in probs:
        logprob = torch.log(prob)
        log_probs += logprob
    log_probs[log_probs < -1000] = 0.
    objective = advantage * log_probs

    actor_loss = objective.mean()
    return actor_loss


class CombinatorialRLModel(BaseModel):

    def __init__(
            self,
            combinatorial_rl_net: CombinatorialRL,
            optimizer: Optional[torch.optim.Optimizer] = None,
            max_grad_norm: Optional[float] = 1,
            learning_rate: Optional[float] = 1e-3,
            device: Optional[str] = "cpu",
            beta: Optional[float] = .9
    ) -> None:
        self.device = device
        self.combinatorial_rl_net = combinatorial_rl_net
        self.combinatorial_rl_net.to(self.device)

        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.combinatorial_rl_net.actor.parameters(), lr=learning_rate
                                              )
        else:
            self.optimizer = optimizer

        self.max_grad_norm = max_grad_norm
        self.beta = beta

    def step(self, episode_number: int, *args) -> Dict:

        inputs = args[0]
        critic_ema = args[1]

        R, probs, actions, actions_idxs = self.combinatorial_rl_net(inputs)

        loss = R.mean()
        if episode_number == 0:
            new_critic_ema = loss
        else:
            new_critic_ema = (critic_ema * self.beta) + \
                             ((1. - self.beta) * loss)

        advantage = R - new_critic_ema
        actor_loss = compute_actor_objective(
            advantage=advantage,
            probs=probs
        )
        self.optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.combinatorial_rl_net.actor.parameters(),
                                       float(self.max_grad_norm), norm_type=2)
        self.optimizer.step()

        return {
            "loss": loss,
            "actor_loss": actor_loss,
            "new_critic_ema": new_critic_ema
        }

    def val_step(self, inputs: torch.autograd.Variable):
        return self.combinatorial_rl_net(inputs)
