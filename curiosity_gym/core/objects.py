"""Object definitions for curiosity-gym grid environments.

This module defines a collection of classes representing various objects for the grid-based
RL environments. Each object can define its own properties, behaviors, and visual representation 
within the environment.
"""

import copy
import random
from abc import ABC, abstractmethod
from typing import Self, override

import numpy as np
import pygame

from curiosity_gym.utils.constants import IX_TO_COLOR, STATE_TO_ROTATION
from curiosity_gym.utils.enums import Action


class GridObject(ABC):
    """Abstract class representing elements that can be placed in a grid environment. \n
    It contains the attributes position, color, and state, which define the characteristics
    and behavior of the object. The class maintains a unique identifier for each subclass 
    and provides default implementations of the :meth:`~reset`, :meth:`~simulate`, 
    :meth:`~get_identity`, :meth:`~is_walkable` and :meth:`~is_harmful` methods. It enforces
    the implementation of a :meth:`~render` method by all subclasses.

    Parameters
    ----------
    position : tuple[int,int]
        Position in the grid where the object will be placed. Values must be in range
        (:attr:`~curiosity_gym.core.gridengine.env_settings.width` - 1, 
        :attr:`~curiosity_gym.core.gridengine.env_settings.height` - 1).
    color : int
        Color of the grid object. Values must be in range(0,10). 
        Color mappings are defined in :const:`~curiosity_gym.utils.constants.IX_TO_COLOR`.
    state : int
        State of the grid object. State characteristics vary by object type. 
        Values must be in range(0,4).
    """

    # Class-level attributes
    identifier = None
    _next_id = 1
    id_map = {}

    def __init__(self, position: tuple[int,int], color: int = 0, state: int = 0) -> None:
        self.start_position = np.array(position)
        self.position = np.array(position)
        self.start_color = color
        self.color = color
        self.start_state = state
        self.state = state

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.identifier = GridObject._next_id
        GridObject.id_map[GridObject._next_id] = cls
        GridObject._next_id += 1

    @abstractmethod
    def render(self, canvas: pygame.Surface, pixelsquare: float):
        """Render grid object in PyGame. \n
        Must be implemented by each grid object type to allow visual recognition
        in *human* rendering mode.

        Parameters
        ----------
        canvas : pygame.Surface
            The PyGame window, in which the grid objects are rendered.
        pixelsquare : float
            The size of a single square in the grid environment.
        """

    def get_identity(self) -> tuple[int|None,int,int]:
        """Return a tuple that uniquely identifies the grid object.

        Returns
        -------
        tuple[int | None, int, int]
            A tuple consisting of :attr:`~identifier`, :attr:`~color` and :attr:`~state`. 
        """
        return (self.__class__.identifier, self.color, self.state)

    def interact(self, agent: Self) -> None:
        """Interact with agent. \n
        Concrete interaction depends on grid object type. The default interaction has no effect.

        Parameters
        ----------
        agent : :class:`~Agent`
            Agent object performing the interaction.
        """

    def reset(self):
        """Reset attributes of the grid object to their starting values. \n
        Sets :attr:`~position`, :attr:`~state` and :attr:`~color` to their respective values 
        assigned when the object was initialised.
        """
        self.position = self.start_position
        self.state = self.start_state
        self.color = self.start_color

    def step(
        self,
        action: Action,
        front_object: Self | None = None,
        walkable: bool = False,
        ) -> float:
        """Compute grid object changes after single timestep. \n
        The default step behaviour is idle and gives 0 reward.

        Parameters
        ----------
        action : :type:`~curiosity_gym.utils.enums.Action`
            Action taken in a grid environment by the RL agent.
        front_object : :type:`GridObject` | None, optional
            Grid object currently in front of the RL agent, by default None.
        walkable : bool, optional
            Wheter the agent can move over the cell in front, by default False.

        Returns
        -------
        float
            Optional reward, which is independent of the environment task and episode termination.
        """
        del action, front_object, walkable
        return 0

    def simulate(
        self, action: Action,
        front_object: Self | None = None,
        walkable: bool = False,
        ) -> Self:
        """Simulate how grid object would change if a given action were taken. \n
        Is used in the :meth:`curiosity_gym.core.gridengine.simulate` method to get the state of 
        the environment if a given action is performed by the RL agent.
        
        Parameters
        ----------
        action : :type:`~curiosity_gym.utils.enums.Action`
            Action of the RL agent to be simulated.
        front_object : :type:`GridObject` | None, optional
            Grid object currently in front of the RL agent, by default None.
        walkable : bool, optional
            Wheter the agent can move over the cell in front, by default False.

        Returns
        -------
        :type:`GridObject`
            Copy of the grid object, including changes caused by the given action.
        """
        simulated = copy.deepcopy(self)
        simulated.step(action, front_object, walkable)
        return simulated

    def is_walkable(self) -> bool:
        """Determine whether agent can move on grid object.\n
        Returns :const:`False` by default. Walkable grid objects must override this method.

        Returns
        -------
        bool
            :const:`True` if agent can move over grid object, :const:`False` otherwise.
        """
        return False

    def is_harmful(self) -> bool:
        """Determine whether grid object is harmful to the agent. \n
        Returns :const:`False` by default. Harmful grid objects must override this method.
        Value is used in :class:`curiosity_gym.core.gridengine` to determine if episode ended.

        Returns
        -------
        bool
            :const:`True` if grid object is harmful, :const:`False` otherwise.
        """
        return False


class Agent(GridObject):
    """Grid object representing the RL agent. \n
    The color of the agent grid object cannot be specified at initialisation, as it is
    used to represent collected :class:`~Key` objects. The agent color will always be 
    initialised as 1 (orange).
    
    Parameters
    ----------
    position : tuple[int,int]
        Position in the grid where the agent object will be placed. Values must be in range
        (:attr:`~curiosity_gym.core.gridengine.env_settings.width` - 1, 
        :attr:`~curiosity_gym.core.gridengine.env_settings.height` - 1).
    state : int
        :type:`Rotation` of the agent.
    """

    @override
    def __init__(self, position: tuple[int,int], state: int = 0) -> None:
        super().__init__(position, 1, state)

    @override
    def step(
        self,
        action: Action,
        front_object: Self | None = None,
        walkable: bool = False,
        ) -> float:
        """Perform a given action. \n
        The agent is the main recipient of the action specified in the step function of
        the environment. For the action of moving forward, the agent considers the walkable 
        parameter provided by the environment to determine if it can change its position 
        accordingly. For the interaction action, the agent will call the :meth:`GridObject.interact`
        method of the object in front of it.

        Parameters
        ----------
        action : :type:`~curiosity_gym.utils.enums.Action`
            Action taken in a grid environment by the RL agent.
        front_object : :type:`GridObject` | None, optional
            Grid object currently in front of the RL agent, by default None.
        walkable : bool, optional
            Wheter the agent can move over the cell in front, by default False.

        Returns
        -------
        float
            Returns 0.
        """

        if action == Action.FORWARD and walkable:
            self.position = self.position + STATE_TO_ROTATION[self.state] * np.array([1,-1])

        elif action == Action.TURN_RIGHT:
            self.state = (self.state - 1) % 4

        elif action == Action.TURN_LEFT:
            self.state = (self.state + 1) % 4

        elif action == Action.INTERACT and front_object:
            front_object.interact(self)

        return 0

    @override
    def render(self, canvas: pygame.Surface, pixelsquare: float) -> None:
        # Constant parameters
        c = np.array([0.5, 0.5])
        d = 0.35
        r = STATE_TO_ROTATION[self.state]

        # Calculate points for triangle rotation
        p1 = (self.position + c - np.array([-d, d]) * r) * pixelsquare
        p2 = (self.position + c - [d, -d] * np.flip(r) + [-d, d] * r ) * pixelsquare
        p3 = (self.position + c + [d, -d] * np.flip(r) + [-d, d] * r ) * pixelsquare

        # Draw agent
        pygame.draw.polygon(
            canvas,
            IX_TO_COLOR[self.color],
            (p1, pygame.Vector2(*p2), pygame.Vector2(*p3)),
            0 # filled triangle
        )

    def get_front(self) -> np.ndarray:
        """Calculate position in front of the agent grid object.

        Returns
        -------
        np.ndarray
            Grid coordinates of the position.
        """
        return self.position + STATE_TO_ROTATION[self.state] * np.array([1,-1])


class Wall(GridObject):
    """Non walkable grid object used to enclose spaces in the environment."""
    @override
    def render(self, canvas: pygame.Surface, pixelsquare: float) -> None:
        pygame.draw.rect(
            canvas,
            IX_TO_COLOR[self.color],
            pygame.Rect(
                pygame.Vector2(*(pixelsquare * self.position)),
                (pixelsquare, pixelsquare),
            ),
        )


class Target(GridObject):
    """Grid object representing a task objective of the environment. \n
    Is used to specify a target location that an agent shoud reach to 
    gain a reward. Can also be used in combination with other objects
    to define more complex task objectives.
    """
    @override
    def render(self, canvas: pygame.Surface, pixelsquare: float) -> None:
        pygame.draw.rect(
            canvas,
            IX_TO_COLOR[self.color],
            pygame.Rect(
                pygame.Vector2(*(pixelsquare * self.position)),
                (pixelsquare, pixelsquare),
            ),
        )

    @override
    def is_walkable(self) -> bool:
        """Target grid objects are always walkable.

        Returns
        -------
        bool
            :const:`True`
        """
        return True


class Door(GridObject):
    """Grid object with walkability depending on its state. \n
    A closed door cannot be walked or seen through. Doors can be opened
    by the agent through interaction. State 2 represents a locked door, 
    that can only be opened if the agent has collected a :class:`Key` 
    object of the same color. \n
    Can be used in combination with :class:`Key` objects to introduce 
    additional reward sparcity to an environment, as more precice 
    actions are required for the agent to get the reward.

    """
    @override
    def interact(self, agent: Agent):
        """Interact with the door.
        The result of the interaction depends on the door state prior to
        the interaction. Closed doors (state = 1) can always be opened 
        by interaction. Locked doors (state = 2) require the prior
        collection of a :class:`Key` object of the same color to open.

        Parameters
        ----------
        agent : :class:`~Agent`
            Agent object performing the interaction.
        """

        if self.state == 2 and agent.color != self.color:
            return
        if self.state == 2 and agent.color == self.color:
            self.state = 0
            agent.color = agent.start_color
        else:
            self.state = (self.state + 1) % 2

    @override
    def render(self, canvas: pygame.Surface, pixelsquare: float) -> None:
        if self.state > 0:
            pygame.draw.rect(
                canvas,
                IX_TO_COLOR[self.color],
                pygame.Rect(
                    pygame.Vector2(*(pixelsquare * self.position)),
                    (pixelsquare, pixelsquare),
                ), 5
            )

            pygame.draw.circle(
                canvas,
                IX_TO_COLOR[self.color],
                pygame.Vector2(*((self.position + np.array([0.75,0.5])) * pixelsquare)),
                0.1 * pixelsquare,
            )

        else:
            pygame.draw.rect(
                canvas,
                IX_TO_COLOR[self.color],
                pygame.Rect(
                    pygame.Vector2(*(pixelsquare * self.position)),
                    (pixelsquare*0.2, pixelsquare),
                ), 5
            )

    @override
    def is_walkable(self) -> bool:
        """Return the walkability of a door, as determined by its state.
        The agent can only walk through the door if it is open.

        Returns
        -------
        bool
            Returns :const:`True` if the state of the door is 0, 
            :const:`False` otherwise.
        """
        return self.state == 0


class Key(GridObject):
    """Collectable grid object that is used to unlock doors of the same color."""

    @override
    def interact(self, agent: Agent) -> None:
        """Collect the key object and remove it from the environment.
        The agent changes color according to the key collected.

        Parameters
        ----------
        agent : :class:`~Agent`
            Agent object performing the interaction.
        """
        agent.color = self.color
        self.position = np.array([-1,-1])

    @override
    def render(self, canvas: pygame.Surface, pixelsquare: float) -> None:
        pygame.draw.circle(
            canvas,
            IX_TO_COLOR[self.color],
            pygame.Vector2(*((self.position + np.array([0.5,0.4])) * pixelsquare)),
            0.15 * pixelsquare,
            5
        )
        pygame.draw.line(
            canvas,
            IX_TO_COLOR[self.color],
            pygame.Vector2(*((self.position + np.array([0.5,0.55])) * pixelsquare)),
            pygame.Vector2(*((self.position + np.array([0.5,0.8])) * pixelsquare)),
            5
        )
        pygame.draw.line(
            canvas,
            IX_TO_COLOR[self.color],
            pygame.Vector2(*((self.position + np.array([0.5,0.7])) * pixelsquare)),
            pygame.Vector2(*((self.position + np.array([0.4,0.7])) * pixelsquare)),
            5
        )


class RandomBlock(GridObject):
    """A non walkable grid object that randomly changes color at every time step. \n
    Can be used to test the `Noisy-TV problem <https://openai.com/index/reinforcement
    -learning-with-prediction-based-rewards/#the-noisy-tv-problem>`__, where certain 
    curiosity-based RL algorithms get stuck when encountering stochastic environment 
    components.
    """

    @override
    def step(
        self,
        action: Action,
        front_object: Self | None = None,
        walkable: bool = False,
        ) -> float:
        """Randomly change grid object color.
        Available colors are taken from :const:`~curiosity_gym.utils.constants.IX_TO_COLOR`.
        """
        self.color = random.randint(0,len(IX_TO_COLOR)-1)
        return 0

    def render(self, canvas: pygame.Surface, pixelsquare: float) -> None:
        pygame.draw.rect(
            canvas,
            IX_TO_COLOR[self.color],
            pygame.Rect(
                pygame.Vector2(*((self.position) * pixelsquare) +
                np.array([0.15,0.15]) * pixelsquare),
                (pixelsquare*0.7, pixelsquare*0.7),
            ),
        )
        font = pygame.font.SysFont("freesansbold", int(pixelsquare))
        img = font.render("?", True, (255, 255, 255))
        canvas.blit(img, pygame.Vector2(*((self.position + np.array([0.275,0.2])) * pixelsquare)))


class Enemy(GridObject):
    """Moving grid object harmful to the agent. \n
    Is used to introduce additional reward sparcity by early episode termination.

    Parameters
    ----------
    position : tuple[int,int]
        Position in the grid where the object will be placed. Values must be in range
        (:attr:`~curiosity_gym.core.gridengine.env_settings.width` - 1, 
        :attr:`~curiosity_gym.core.gridengine.env_settings.height` - 1).
    state : int
        State of the enemy grid object. Determines current movement direction.
    reach : int
        Number of cells the enemy can move from its starting position.
    """
    @override
    def __init__(self,
                 position: tuple[int,int],
                 state: int = 0,
                 reach: int = 2
                 ) -> None:
        super().__init__(position, 9, state)
        self.reach = reach

    @override
    def is_harmful(self) -> bool:
        """The enemy grid object is always harmful to the agent.\n
        This will terminate the episode if agent and an enemy are
        on the same grid position.
        Returns
        -------
        bool
            :const:`True`
        """
        return True

    @override
    def step(
        self,
        action: Action,
        front_object: Self | None = None,
        walkable: bool = False,
        ) -> float:
        """Perform enemy movement.
        Behaviour is determined by starting position and reach of the enemy grid object.

        Parameters
        ----------
        action : :type:`~curiosity_gym.utils.enums.Action`
            Action taken in a grid environment by the RL agent.
        front_object : :type:`GridObject` | None, optional
            Grid object currently in front of the RL agent, by default None.
        walkable : bool, optional
            Wheter the agent can move over the cell in front, by default False.

        Returns
        -------
        float
            Returns 0.
        """
        self.position = self.position + STATE_TO_ROTATION[self.state] * np.array([1,-1])
        if self.position[self.state%2] == self.start_position[self.state%2] + self.reach or \
        self.position[self.state%2] == self.start_position[self.state%2] - self.reach or \
        self.position[self.state%2] == self.start_position[self.state%2]:
            self.state = (self.state + 2) % 4
        return 0

    @override
    def is_walkable(self) -> bool:
        """The enemy grid object is always walkable. 
        Allows for episode termination by enemy contact.

        Returns
        -------
        bool
            :const:`True`
        """
        return True

    @override
    def render(self, canvas: pygame.Surface, pixelsquare: float) -> None:
        pygame.draw.polygon(
            canvas,
            IX_TO_COLOR[self.color],
            (
                pygame.Vector2(*((self.position + np.array([0.25,0.2])) * pixelsquare)),
                pygame.Vector2(*((self.position + np.array([0.75,0.2])) * pixelsquare)),
                pygame.Vector2(*((self.position + np.array([0.85,0.55])) * pixelsquare)),
                pygame.Vector2(*((self.position + np.array([0.5,0.85])) * pixelsquare)),
                pygame.Vector2(*((self.position + np.array([0.15,0.55])) * pixelsquare)),
            )
        )
        pygame.draw.circle(
            canvas,
            (255,255,255),
            pygame.Vector2(*((self.position + np.array([0.35,0.37])) * pixelsquare)),
            5
        )
        pygame.draw.circle(
            canvas,
            (255,255,255),
            pygame.Vector2(*((self.position + np.array([0.65,0.37])) * pixelsquare)),
            5
        )
        pygame.draw.line(
            canvas,
            (255,255,255),
            pygame.Vector2(*((self.position + np.array([0.4,0.63])) * pixelsquare)),
            pygame.Vector2(*((self.position + np.array([0.6,0.63])) * pixelsquare)),
            3
        )


class SmallReward(GridObject):
    """Grid object yielding a small reward when agent moves over it.

    Parameters
    ----------
    position : tuple[int,int]
        Position in the grid where the object will be placed. Values must be in range
        (:attr:`~curiosity_gym.core.gridengine.env_settings.width` - 1, 
        :attr:`~curiosity_gym.core.gridengine.env_settings.height` - 1).
    reward : float
        Amount of reward to yield when agent walks over the grid object.
    """

    @override
    def __init__(self,
                 position: tuple[int,int],
                 reward: float,
                 ) -> None:
        super().__init__(position, 9, 0)
        self.reward = reward

    @override
    def is_walkable(self) -> bool:
        """SmallReward grid object is always walkable.
        Allows for the agent to collect the reward by walking over it.

        Returns
        -------
        bool
            :const:`True`
        """
        return True

    @override
    def step(
        self,
        action: Action,
        front_object: Self | None = None,
        walkable: bool = False,
        ) -> float:
        """Determine whether agent is walking over the grid object.

        Parameters
        ----------
        action : :type:`~curiosity_gym.utils.enums.Action`
            Action taken in a grid environment by the RL agent.
        front_object : :type:`GridObject` | None, optional
            Grid object currently in front of the RL agent, by default None.
        walkable : bool, optional
            Wheter the agent can move over the cell in front, by default False.

        Returns
        -------
        float
            Returns :attr:`SmallReward.reward` if agent is moving over small reward
            object, 0 otherwise.
        """
        if front_object == self:
            self.position = np.array([-1,-1])
            return self.reward
        return 0

    @override
    def render(self, canvas: pygame.Surface, pixelsquare: float) -> None:
        pygame.draw.polygon(
        canvas,
        IX_TO_COLOR[self.color],
        (
            pygame.Vector2(*((self.position + np.array([0.5, 0.15])) * pixelsquare)),
            pygame.Vector2(*((self.position + np.array([0.65, 0.35])) * pixelsquare)),
            pygame.Vector2(*((self.position + np.array([0.65, 0.65])) * pixelsquare)),
            pygame.Vector2(*((self.position + np.array([0.5, 0.85])) * pixelsquare)),
            pygame.Vector2(*((self.position + np.array([0.35, 0.65])) * pixelsquare)),
            pygame.Vector2(*((self.position + np.array([0.35, 0.35])) * pixelsquare)),
        )
    )


class Ball(GridObject):
    """Non walkable grid object that can be moved by agent interaction. \n
    Can be used to define more complex task objectives that differ from
    navigation tasks.

    Parameters
    ----------
    position : tuple[int,int]
        Position in the grid where the object will be placed. Values must be in range
        (:attr:`~curiosity_gym.core.gridengine.env_settings.width` - 1, 
        :attr:`~curiosity_gym.core.gridengine.env_settings.height` - 1).
    zone_low : tuple[int,int]
        Low boundaries of the zone in which the ball grid object can be moved.
    zone_high: tuple[int,int]
        High boundaries of the zone in which the ball grid object can be moved.
    color : int
        Color of the grid object.
    """

    @override
    def __init__(self,
                 position: tuple[int,int],
                 zone_low: tuple[int,int],
                 zone_high: tuple[int,int],
                 color: int = 9,
                 ) -> None:
        super().__init__(position, color, 0)
        self.zone_low = zone_low
        self.zone_high = zone_high

    @override
    def interact(self, agent: Agent) -> None:
        """Move ball one cell in direction of agent rotation.
        Only works if new location is in zone defined by :attr:`Ball.zone_low`,
        :attr:`Ball.zone_high`.

        Parameters
        ----------
        agent : :class:`~Agent`
            Agent object performing the interaction.
        """
        pos_new = self.position + self.position - agent.position
        if (self.zone_low[0] <= pos_new[0] <= self.zone_high[0]) and (
            self.zone_low[1] <= pos_new[1] <= self.zone_high[1]):
            self.position = pos_new

    @override
    def render(self, canvas: pygame.Surface, pixelsquare: float) -> None:
        pygame.draw.circle(
            canvas,
            IX_TO_COLOR[self.color],
            pygame.Vector2(*((self.position + np.array([0.5,0.5])) * pixelsquare)),
            0.35 * pixelsquare,
            0
        )
        pygame.draw.circle(
            canvas,
            IX_TO_COLOR[0],
            pygame.Vector2(*((self.position + np.array([0.5,0.5])) * pixelsquare)),
            0.36 * pixelsquare,
            3
        )
