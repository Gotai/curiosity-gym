from typing_extensions import override
from curiosity_gym.core.objects.grid_object import GridObject

from curiosity_gym.utils.constants import IX_TO_COLOR
import pygame

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
