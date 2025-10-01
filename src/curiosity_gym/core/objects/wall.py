from typing_extensions import override
from curiosity_gym.core.objects.grid_object import GridObject

from curiosity_gym.utils.constants import IX_TO_COLOR
import pygame

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
