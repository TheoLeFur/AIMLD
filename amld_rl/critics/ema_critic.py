import torch
from typing import Optional, Dict

from amld_rl.critics.base_critic import BaseCritic


class EMACritic(BaseCritic):

    def __init__(
            self,
            beta: Optional[float] = .9,
            device: Optional[str] = "cpu"
    ):
        """
        Exponential moving average of the rewards used as a simple baseline for Policy Gradients
        Algorithm

        Args:
            beta: decay coefficient for EMA
        """
        super().__init__()

        self.beta = beta
        self.ema: torch.Tensor = torch.zeros(1).to(device)

    def update_critic(self, inputs: torch.Tensor, *args) -> Dict[str, torch.Tensor]:
        """
        Args:
            inputs:
            *args: args[0] contains [L(pi|s), ... , L(pi|s)],
            where the vector has length BATCH_SIZE, args[1] contains the episode number
        Returns: Updated value of the EMA
        """

        R: torch.Tensor = args[0]
        episode_number: int = args[1]
        loss_value: torch.Tensor = R.mean()

        if episode_number == 0:
            self.ema = loss_value
        else:
            self.ema = self.beta * self.ema + (1 - self.beta) * loss_value

        return {"baseline_pred": self.ema}
