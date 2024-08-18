from abc import ABC, abstractmethod
from typing import override

import numpy as np
from gymnasium import spaces

from curiosity_gym.core import objects


class AgentPOV(ABC):

    def __init__(
            self,
            action_space: spaces.Discrete,
            observation_space: spaces.MultiDiscrete,
            width: int,
            height: int,
        ) -> None:
        self.action_space = action_space
        self.observation_space = observation_space
        self.width = width
        self.height = height
        self.visible_positions = []

    @abstractmethod
    def transform_obs(self, state: np.ndarray, agent: objects.Agent) -> np.ndarray:
        pass

    def transform_action(self, action):
        return action

    def is_visible(self, state, pos_agent, pos_cell):
        dx = np.sign(pos_cell[0] - pos_agent[0])
        dy = np.sign(pos_cell[1] - pos_agent[1])
        x, y = pos_agent[0] + dx, pos_agent[1] + dy

        while (x, y) != (pos_cell[0], pos_cell[1]):
            if state[x + y * self.width][0] == objects.Wall.identifier or \
                state[x + y * self.width][0] == objects.Door.identifier and \
                state[x + y * self.width][2] == 2:
                return False

            x += dx if x != pos_cell[0] else 0
            y += dy if y != pos_cell[1] else 0
        return True


class GlobalView(AgentPOV):

    @override
    def __init__(self, width: int, height: int) -> None:
        action_space = spaces.Discrete(4)
        number_of_nodes = width * height
        observation_space = spaces.MultiDiscrete(np.full((number_of_nodes,3), 10, dtype=int))
        super().__init__(action_space, observation_space, width, height)

    @override
    def transform_obs(self, state: np.ndarray, agent: objects.Agent) -> np.ndarray:
        return state


class LocalView(AgentPOV):

    @override
    def __init__(self, radius: int, width: int, height: int) -> None:
        self.radius = radius
        action_space = spaces.Discrete(4)
        number_of_cells = (self.radius * 2 + 1)**2
        observation_space = spaces.MultiDiscrete(np.full((number_of_cells,3), 10, dtype=int))
        super().__init__(action_space, observation_space, width, height)

    @override
    def transform_obs(self, state: np.ndarray, agent: objects.Agent) -> np.ndarray:
        self.visible_positions = []
        pos = agent.position
        local = np.full(((self.radius * 2 + 1)**2,3), [0,0,0])
        for x in range(-self.radius, self.radius + 1):
            for y in range(-self.radius, self.radius + 1):
                ix = (pos[0] + x) + self.width * (pos[1] + y)
                ix_new = self.radius + x + (self.radius * 2 + 1) * (self.radius + y)
                cell = (pos[0] + x, pos[1] + y)

                if ix < 0 or not self.is_visible(state, pos, cell):
                    continue

                self.visible_positions.append(cell)
                local[ix_new] = state[ix]
        return local


class ForwardView(AgentPOV):

    @override
    def __init__(self, pov_length: int, pov_width: int, width: int, height: int) -> None:
        self.pov_width = pov_width
        self.pov_length = pov_length
        action_space = spaces.Discrete(4)
        number_of_cells = (pov_length + 1) * pov_width
        observation_space = spaces.MultiDiscrete(np.full((number_of_cells,3), 10, dtype=int))
        super().__init__(action_space, observation_space, width, height)

    @override
    def transform_obs(self, state: np.ndarray, agent: objects.Agent) -> np.ndarray:
        self.visible_positions = []
        pos = agent.position
        local = np.full(((self.pov_length + 1) * self.pov_width,3), [0,0,0])

        if agent.state % 3 == 0:
            length_range = range(0, self.pov_length + 1)
            width_range = range(-int(self.pov_width/2), int(self.pov_width/2) + 1)

        else:
            length_range = range(0, -self.pov_length - 1, -1)
            width_range = range(int(self.pov_width/2), -int(self.pov_width/2) - 1, -1)

        ix_new = 0
        for l in length_range:
            for w in width_range:
                if agent.state % 2 == 0:
                    x, y = l, w
                else:
                    x, y = w, l

                ix = (pos[0] + x) + self.width * (pos[1] + y)
                cell = (pos[0] + x, pos[1] + y)

                if ix < 0 or not self.is_visible(state, pos, cell):
                    continue

                self.visible_positions.append(cell)
                local[ix_new] = state[ix]
                ix_new += 1

        return local
