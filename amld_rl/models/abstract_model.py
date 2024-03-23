from abc import ABC, abstractmethod


class BaseModel(ABC):

    @abstractmethod
    def step(self, episode_number: int, *args, **kwargs) -> None:
        raise NotImplementedError
