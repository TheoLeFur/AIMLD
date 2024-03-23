from typing import List

import numpy as np

from amld_rl.envs.easy21 import *
from amld_rl.models.abstract_model import BaseModel


class SarsaModel(BaseModel):

    def __init__(self, params: dict) -> None:

        self.N0: int = params["N0"]
        self.environment: Easy21 = params["environment"]
        self.lambda_param: float = params["lambda_param"]
        self.gamma: float = params["gamma"]

        self.V = np.zeros(shape=(
            self.environment.dealer_value_count, self.environment.player_value_count))
        self.Q = np.zeros(shape=(
            self.environment.dealer_value_count, self.environment.player_value_count, self.environment.action_count))
        self.N = np.zeros(shape=(
            self.environment.dealer_value_count, self.environment.player_value_count, self.environment.action_count))
        self.E = np.zeros(shape=(
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

        epsilon: float = self.N0 / \
                         (self.N0 +
                          np.sum(self.N[state.dealer - 1, state.player - 1, :]))

        if random.random() < epsilon:
            if random.random() < 0.5:
                action: Action = Action.hit
            else:
                action: Action = Action.stick
        else:
            action = Action.to_action(np.argmax(self.Q[state.dealer - 1, state.player - 1, :]))

        return action

    def step(self, episode_number: int, *args) -> None:
        # random start
        s: State = self.environment.reset()
        a: Action = self.choose_action(s)

        while not s.terminal:
            # update N(s,a)
            self.N[s.dealer - 1, s.player - 1, Action.to_int(a)] += 1
            # execute action a and observe s_new, r
            s_new, r = self.environment.step(s, a)
            dealer_id = s.dealer - 1
            player_id = s.player - 1
            if s_new.terminal:
                Q_new = 0
            else:
                a_new: Action = self.choose_action(s_new)
                dealer_id_new = s_new.dealer - 1
                player_id_new = s_new.player - 1
                Q_new = self.Q[dealer_id_new, player_id_new, Action.to_int(a_new)]
            alpha = 1.0 / self.N[dealer_id, player_id, Action.to_int(a)]
            td_error = r + self.gamma * Q_new - self.Q[dealer_id, player_id, Action.to_int(a)]
            self.E[dealer_id, player_id, Action.to_int(a)] += 1
            self.Q += alpha * td_error * self.E
            self.E *= self.gamma * self.lambda_param
            s: State = s_new
            if not s_new.terminal:
                a: Action = a_new
        self.update_value()

    def update_value(self) -> None:
        self.V = np.max(self.Q, axis=2)
