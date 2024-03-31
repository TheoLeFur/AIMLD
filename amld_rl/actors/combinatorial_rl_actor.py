from typing import Callable, Optional, Tuple, List

import numpy as np
import torch
from torch import nn as nn
import os

from amld_rl.neural_nets.pointer_net import PointerNet
from amld_rl.actors.base_actor import BaseActor


class CombinatorialRLActor(BaseActor):
    def __init__(self,
                 embedding_size: int,
                 hidden_dim:
                 int,
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
            n_samples: Optional[int] = int(1e4)
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
            self.eval()
            R, action_probs, actions, action_indices = self(
                input_graph.unsqueeze(0))
            if R < best_reward:
                best_reward = R
                best_action = actions

        return best_reward, best_action

    def save_weights(self, epoch: int, path: str, exp_name: str = "1e5DS") -> None:
        """

        Args:
            epoch: Epoch Number
            path: Name of path for saving the weights
            exp_name: Name of the experiment

        Returns: None

        """
        if not os.path.exists(path):
            os.makedirs(path)
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

        action_probs: List = []
        for prob, action_id in zip(probs, action_idxs):
            action_probs.append(prob[torch.tensor(
                [x for x in range(batch_size)], device=self.device), action_id.data])

        R = self.reward(actions, self.device)

        return R, action_probs, actions, action_idxs
