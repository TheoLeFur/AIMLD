from amld_rl.trainer.base_trainer import BaseTrainer
from typing import Dict
from amld_rl.models.abstract_model import BaseModel
from tqdm import tqdm


class Trainer(BaseTrainer):

    def __init__(self, model: BaseModel, n_episodes):
        self.n_episodes = n_episodes
        self.model = model

    def train(self) -> None:
        for episode in tqdm(range(self.n_episodes)):
            self.model.step(episode_number=episode)
