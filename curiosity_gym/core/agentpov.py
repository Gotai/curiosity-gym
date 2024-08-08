from abc import ABC, abstractmethod
from typing import override

import numpy as np
from gymnasium import spaces

from core import objects


class AgentPOV(ABC):

    def __init__(self, action_space: spaces.Discrete, observation_space: spaces.Box) -> None:
        self.action_space = action_space
        self.observation_space = observation_space

    @abstractmethod
    def transform_obs(self, state: np.ndarray, agent: objects.Agent) -> np.ndarray:
        pass

    def transform_action(self, action):
        return action


class FullView(AgentPOV):

    @override
    def __init__(self) -> None:
        action_space = spaces.Discrete(4)
        observation_space = spaces.Box(0, 7, (100,3), dtype=np.float64)
        super().__init__(action_space, observation_space)

    @override
    def transform_obs(self, state: np.ndarray, agent: objects.Agent) -> np.ndarray:
        return state
