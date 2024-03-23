from typing import List

import numpy as np

from amld_rl.envs.easy21 import *
from amld_rl.models.abstract_model import BaseModel


class MCModel(BaseModel):
    def __init__(self, params: dict) -> None:

        self.N0: int = params["N0"]
        self.environment: Easy21 = params["environment"]
        self.episode_log_freq: int = params["episode_log_freq"]

        self.V: np.ndarray = np.zeros(shape=(self.environment.dealer_value_count, self.environment.player_value_count))

        self.Q: np.ndarray = np.zeros(shape=(
            self.environment.dealer_value_count, self.environment.player_value_count, self.environment.action_count))

        self.N: np.ndarray = np.zeros(shape=(
            self.environment.dealer_value_count, self.environment.player_value_count, self.environment.action_count))

        self.returns: List = []
        for _ in range(self.environment.dealer_value_count * self.environment.player_value_count):
            G: List = []
            for _ in range(self.environment.action_count):
                G.append([])
            self.returns.append(G)

        self.count_wins: int = 0
        self.episodes: int = 0

    @property
    def get_state_value(self) -> np.ndarray:
        return self.V

    @property
    def get_state_action_value(self) -> np.ndarray:
        return self.Q

    def choose_action(self, state: State) -> Action:

        """
        Selects action based on current state using an epsilon greedy scheme.
        With probability less than epsilon, we select an action randomly from the environment,
        else we select the action greedily, according to the Bellman backup update.
        """

        epsilon = self.N0 / (self.N0 + np.sum(self.N[state.dealer - 1, state.player - 1, :]))
        if random.random() < epsilon:
            if random.random() < 0.5:
                action = Action.hit
            else:
                action = Action.stick
        else:
            action = Action.to_action(np.argmax(self.Q[state.dealer - 1, state.player - 1, :]))
        return action

    def step(self, episode_number: int, *args):

        """
        Training loop for
        """

        state: State = self.environment.reset()
        episode_pairs: List = []
        score: int = 0
        reward = 0

        while not state.terminal:
            action: Action = self.choose_action(state)
            self.N[state.dealer - 1, state.player - 1, Action.to_int(action)] += 1
            episode_pairs.append((state, action))
            state, reward = self.environment.step(state, action)
            score += reward

        for state, action in episode_pairs:
            idx: int = self.environment.dealer_value_count * (state.dealer - 1) + state.player
            self.returns[idx][Action.to_int(action)].append(reward)
            error = np.mean(self.returns[idx][Action.to_int(action)] - self.Q[
                state.dealer - 1, state.player - 1, Action.to_int(action)])
            alpha = 1. / self.N[state.dealer - 1, state.player - 1, Action.to_int(action)]
            self.Q[state.dealer - 1, state.player - 1, Action.to_int(action)] += alpha * error
            self.V[state.dealer - 1, state.player - 1] = np.max(self.Q[state.dealer - 1, state.player - 1, :])
