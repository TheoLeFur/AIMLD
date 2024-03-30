from typing import Optional, List, Dict

import torch

from amld_rl.actors.combinatorial_rl_actor import CombinatorialRLActor
from amld_rl.models.abstract_model import BaseModel


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
            combinatorial_rl_net: CombinatorialRLActor,
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
