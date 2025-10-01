from typing_extensions import override
from curiosity_gym.core.pov import AgentPOV

import numpy as np
from gymnasium import spaces
from curiosity_gym.core.objects import Agent

class LocalView(AgentPOV):
    """Agent point-of-view observing grid cells in a given radius around the agent.

    Parameters
    ----------
    radius : int
        Number of cells around the agent to be part of the observation.
    env_size : tuple[int, int]
        Number of (horizontal, vertical) cells in the grid environment where the pov class is used.
    xray : bool
        Whether the agent can observe cells behind walls and closed doors.


    .. figure:: ../../source/images/LocalView_2.gif
        :width: 500
        :align: center

        Example of LocalView with a radius of 2 without xray.
    """

    @override
    def __init__(
        self, radius: int, env_size: tuple[int, int], xray: bool = False
    ) -> None:
        self.radius = radius
        self.xray = xray
        action_space = spaces.Discrete(4)
        number_of_cells = (self.radius * 2 + 1) ** 2
        observation_space = spaces.Box(
            shape=(number_of_cells, 3), high=10, low=0, dtype=np.int64
        )
        super().__init__(action_space, observation_space, env_size)

    @override
    def transform_obs(self, state: np.ndarray, agent: Agent) -> np.ndarray:
        self.visible_positions = []
        pos = agent.position.tolist()
        local = np.full(((self.radius * 2 + 1) ** 2, 3), [0, 0, 0])
        for x in range(-self.radius, self.radius + 1):
            for y in range(-self.radius, self.radius + 1):
                ix = (pos[0] + x) + self.width * (pos[1] + y)
                ix_new = self.radius + x + (self.radius * 2 + 1) * (self.radius + y)
                cell = (pos[0] + x, pos[1] + y)

                if ix < 0 or (not self.is_visible(state, pos, cell) and not self.xray):
                    continue

                self.visible_positions.append(cell)
                local[ix_new] = state[ix]
        return local
