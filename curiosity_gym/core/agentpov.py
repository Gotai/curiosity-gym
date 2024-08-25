"""Definitions for inputs and outputs of the curiosity-gym grid environments.

This module defines point-of-view classes that implement the processing of 
inputs going into a grid environment (actions) and outputs coming out of a
grid environment (observations). By default, there are three distinct pov
classes. The GlobalView always returns the full state of an environment.
The observation of LocalView only contains grid cells that are inside a
given radius of the agent. The ForwardView only contains grid cells that
are in front of the agent within a field of given width and length. 
LocalView and ForwardView only provide information about a cell if there 
is no wall or closed door between the agent and the particular cell.
"""

from abc import ABC, abstractmethod
from typing import override

import numpy as np
from gymnasium import spaces

from curiosity_gym.core import objects
from curiosity_gym.utils.enums import Action


class AgentPOV(ABC):
    """Abstract agent point-of-view class. \n
    Implements the processing of inputs going into a grid environment (actions)
    and outputs coming out of a grid environment (observations). The :meth:`transform_obs`
    method has to be implemented by inheriting pov classes. 

    Parameters
    ----------
    action_space : gymnasium.spaces.Discrete
        The action space defining what actions can be taken by an RL agent within a grid 
        environment.
    observation_space : gymnasium.spaces.MultiDiscrete
        The observation space defining the structure of the observations being returned
        by a grid environment.
    width : int
        Number of horizontal cells in the grid environment where the pov class is used.
    height : int
        Number of vertical cells in the grid environment where the pov class is used.
    """

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
        """Transform environment state to agent observation.

        Parameters
        ----------
        state : np.ndarray
            State provided by the grid environment.
        agent : objects.Agent
            Agent grid object within the grid environment. Is required to construct
            observations in relation to the agent objects position and rotation.

        Returns
        -------
        np.ndarray
            Observation within the :attr:`~observation_space`.
        """

    def transform_action(self, action: int | Action) -> Action:
        """Transform given action so that it is compatible with the grid environment. \n
        Can be used to define alternative action spaces and map them to the grid dynamics.

        Parameters
        ----------
        action : int | :type:`~curiosity_gym.utils.enums.Action`
            Action selected by the RL agent.

        Returns
        -------
        :type:`~curiosity_gym.utils.enums.Action`
            Action within :attr:`action_space` to perform.
        """
        return Action(action)

    def is_visible(
            self,
            state: np.ndarray,
            pos_agent: tuple[int,int],
            pos_cell: tuple[int,int],
        ) -> bool:
        """Check if grid cell at given position is visible by the agent. \n
        A grid cell is visible if there is no wall or closed door between the agent and the cell.

        Parameters
        ----------
        state : np.ndarray
            Current environment state.
        pos_agent : tuple[int,int]
            Position of the agent grid object.
        pos_cell : tuple[int,int]
            Position of the cell to check for visibility.

        Returns
        -------
        bool
            True if there is no wall or closed door between agent and the given cell,
            False otherwise.
        """
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
    """Agent point-of-view observing the full state of the environment.

    Parameters
    ----------
    width : int
        Number of horizontal cells in the grid environment where the pov class is used.
    height : int
        Number of vertical cells in the grid environment where the pov class is used.
    """

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
    """Agent point-of-view observing grid cells in a given radius around the agent.

    Parameters
    ----------
    radius : int
        Number of cells around the agent to be part of the observation.
    width : int
        Number of horizontal cells in the grid environment where the pov class is used.
    height : int
        Number of vertical cells in the grid environment where the pov class is used.

    
    .. figure:: ../../source/images/LocalView_2.gif
        :width: 500
        :align: center
        
        Example of LocalView with a radius of 2.
    """

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
        pos = agent.position.tolist()
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
    """Agent point-of-view observing grid cells in front of the agent.

    Parameters
    ----------
    pov_length : int
        Length of the visible field in front of the agent.
    pov_width : int
        Width of the visible field in front of the agent.
    env_width : int
        Number of horizontal cells in the grid environment where the pov class is used.
    env_height : int
        Number of vertical cells in the grid environment where the pov class is used.

    
    .. figure:: ../../source/images/ForwardView_2_3.gif
        :width: 500
        :align: center
        
        Example of ForwardView with a length of 2 and a width of 3.
    """

    @override
    def __init__(self, pov_length: int, pov_width: int, env_width: int, env_height: int) -> None:
        self.pov_width = pov_width
        self.pov_length = pov_length
        action_space = spaces.Discrete(4)
        number_of_cells = (pov_length + 1) * pov_width
        observation_space = spaces.MultiDiscrete(np.full((number_of_cells,3), 10, dtype=int))
        super().__init__(action_space, observation_space, env_width, env_height)

    @override
    def transform_obs(self, state: np.ndarray, agent: objects.Agent) -> np.ndarray:
        self.visible_positions = []
        pos = agent.position.tolist()
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
