from dataclasses import dataclass, field

import numpy as np
import pygame

import objects


@dataclass
class EnvironmentSettings:
    max_steps: int = 50
    width: int = 10
    height: int = 10
    reward_range: tuple[int,int] = (0,1)


@dataclass
class RenderSettings:
    render_mode: str | None = None
    render_fps: int = 4
    window_width: int = 512
    window_height: int = 512
    window: pygame.surface.Surface | None = None
    clock: pygame.time.Clock | None = None


@dataclass
class EnvironmentObjects:
    agent: objects.Agent
    target: objects.Target
    walls: np.ndarray = field(default_factory = lambda: np.empty(0, dtype = objects.Wall))
    other: np.ndarray = field(default_factory = lambda: np.empty(0, dtype = objects.GridObject))

    def get_all(self) -> np.ndarray:
        return np.concatenate((self.get_non_wall(), self.walls))

    def get_non_wall(self) -> np.ndarray:
        return np.concatenate((self.other, np.array([self.target]), np.array([self.agent])))
