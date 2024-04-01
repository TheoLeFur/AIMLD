from typing import Optional, List, Dict

import torch

from amld_rl.actors.combinatorial_rl_actor import CombinatorialRLActor
from amld_rl.models.abstract_model import BaseModel
from amld_rl.critics.base_critic import BaseCritic


def compute_actor_objective(
        advantage: torch.Tensor,
        probs: List
) -> torch.Tensor:
    """

    Args:
        advantage:
        probs:

    Returns:

    """
    log_probs = 0
    for prob in probs:
        logprob = torch.log(prob)
        log_probs += logprob
    log_probs[log_probs < -1000] = 0.
    objective = advantage * log_probs

    actor_loss = objective.mean().detach().clone().requires_grad_(True)
    return actor_loss


class CombinatorialRLModel(BaseModel):

    def __init__(
            self,
            combinatorial_rl_net: CombinatorialRLActor,
            combinatorial_rl_critic: BaseCritic,
            optimizer: Optional[torch.optim.Optimizer] = None,
            max_grad_norm: Optional[float] = 1,
            learning_rate: Optional[float] = 1e-3,
            device: Optional[str] = "cpu",
    ) -> None:

        """

        Args:
            combinatorial_rl_net:
            combinatorial_rl_critic:
            optimizer:
            max_grad_norm:
            learning_rate:
            device:
            beta:
        """
        self.device = device

        self.combinatorial_rl_net = combinatorial_rl_net
        self.combinatorial_rl_critic = combinatorial_rl_critic
        self.combinatorial_rl_net.to(self.device)
        self.combinatorial_rl_critic.to(self.device)

        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.combinatorial_rl_net.actor.parameters(), lr=learning_rate
                                              )
        else:
            self.optimizer = optimizer

        self.max_grad_norm = max_grad_norm

    def _get_baseline(self, inputs: torch.Tensor, R: torch.Tensor, episode_number: int):
        """

        Args:
            inputs: Input Tensor
            R: Tensor of rewards

        Returns: Prediction for next baseline

        """

        critic_data: Dict = self.combinatorial_rl_critic.update_critic(
            inputs,
            R,
            episode_number
        )
        return critic_data

    def step(self, episode_number: int, *args) -> Dict[str, torch.Tensor]:

        inputs: torch.Tensor = args[0]
        R, probs, actions, actions_idxs = self.combinatorial_rl_net(inputs)

        loss: torch.Tensor = R.mean()
        critic_data: Dict[str, torch.Tensor] = self._get_baseline(
            inputs=inputs,
            R=R,
            episode_number=episode_number
        )
        baseline_pred: torch.Tensor = critic_data["baseline_pred"]

        advantage: torch.Tensor = R - baseline_pred
        actor_loss: torch.Tensor = compute_actor_objective(
            advantage=advantage,
            probs=probs
        )

        self.optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.combinatorial_rl_net.actor.parameters(),
                                       float(self.max_grad_norm), norm_type=2)
        with torch.no_grad():
            self.optimizer.step()

        return {
            "loss": loss,
            "actor_loss": actor_loss,
            "new_critic_ema": baseline_pred,
            "critic_loss": critic_data["critic_loss"]
        }

    def val_step(self, inputs: torch.Tensor):
        return self.combinatorial_rl_net(inputs)
