from abc import ABC, abstractmethod
from typing import Any


class BaseState(ABC):
    def __init__(self, discrete: bool) -> None:
        self.discrete = discrete


class BaseEnv(ABC):

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def step(self, state: BaseState, action: Any):
        raise NotImplementedError
