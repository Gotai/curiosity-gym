from typing_extensions import override
from curiosity_gym.core.objects.grid_object import GridObject

import numpy as np
from curiosity_gym.utils.constants import IX_TO_COLOR
import pygame

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
    def __init__(
        self,
        position: tuple[int, int],
        zone_low: tuple[int, int],
        zone_high: tuple[int, int],
        color: int = 9,
    ) -> None:
        super().__init__(position, color, 0)
        self.zone_low = zone_low
        self.zone_high = zone_high

    @override
    def interact(self, agent: GridObject) -> None:
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
            self.zone_low[1] <= pos_new[1] <= self.zone_high[1]
        ):
            self.position = pos_new

    @override
    def render(self, canvas: pygame.Surface, pixelsquare: float) -> None:
        pygame.draw.circle(
            canvas,
            IX_TO_COLOR[self.color],
            pygame.Vector2(*((self.position + np.array([0.5, 0.5])) * pixelsquare)),
            0.35 * pixelsquare,
            0,
        )
        pygame.draw.circle(
            canvas,
            IX_TO_COLOR[0],
            pygame.Vector2(*((self.position + np.array([0.5, 0.5])) * pixelsquare)),
            0.36 * pixelsquare,
            3,
        )
