from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
import numpy as np
import pygame

import objects
from agentpov import AgentPOV
from enums import Action
from utils import EnvironmentSettings, RenderSettings, EnvironmentObjects


class GridEnv(gym.Env, ABC):
    """Placeholder"""

    def __init__(
        self,
        agent_pov: AgentPOV,
        env_settings: EnvironmentSettings,
        render_settings: RenderSettings,
        env_objects: EnvironmentObjects,
    ) -> None:

        # Get Action & Observations spaces
        self.agent_pov = agent_pov
        self.action_space = agent_pov.action_space
        self.observation_space = agent_pov.observation_space

        # Store Settings
        self.env_settings = env_settings
        self.render_settings = render_settings

        # Current Environment State
        self.objects = env_objects
        self.step_count = 0

        # Initialise Render Objects
        if self.render_settings.render_mode == "human":
            pygame.init()
            pygame.display.init()
            window_size = (self.render_settings.window_width, self.render_settings.window_height)
            self.render_settings.window = pygame.display.set_mode(window_size)
            self.render_settings.clock = pygame.time.Clock()

    @abstractmethod
    def _get_reward(self) -> int:
        pass

    @abstractmethod
    def _get_terminated(self) -> bool:
        return self.step_count >= self.env_settings.max_steps

    def step(self, action: int | Action) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        for ob in self.objects.get_non_wall():
            ob.step(
                Action(action),
                self._find_object(self.objects.agent.get_front()),
                self._check_walkable(self.objects.agent.get_front())
                )

        if self.render_settings.render_mode == "human":
            self._render_frame()

        self.step_count += 1
        return self._get_obs(), self._get_reward(), self._get_terminated(), False, self._get_info()

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        super().reset(**kwargs)
        for ob in list(self.objects.other) + [self.objects.agent, self.objects.target]:
            ob.reset()
        self.step_count = 0
        return (self._get_obs(), self._get_info())

    def render(self) -> np.ndarray | None:
        if self.render_settings.render_mode == "rgb_array":
            return self._render_frame()
        return None

    def close(self) -> None:
        if self.render_settings.window is not None:
            pygame.display.quit()
            pygame.quit()
    
    def get_object_ids(self) -> dict[objects.GridObject, int]:
        return objects.GridObject.id_map

    def get_state(self) -> np.ndarray:
        state = np.zeros([self.env_settings.width * self.env_settings.height, 3])
        for ob in self.objects.get_all():
            x,y = ob.position
            state[x + y * self.env_settings.width] = ob.get_identity()
        return state

    def load_walls(self, positions: np.ndarray) -> np.ndarray:
        walls = [objects.Wall(position) for position in positions]
        return np.array(walls)

    def simulate(self, action: int | Action) -> np.ndarray:
        state = np.zeros([self.env_settings.width * self.env_settings.height, 3])
        for ob in self.objects.get_all():
            ob_simulated = ob.simulate(
                Action(action),
                self._find_object(self.objects.agent.get_front()),
                self._check_walkable(self.objects.agent.get_front())
                )
            x,y = ob_simulated.position
            state[x + y * self.env_settings.width] = ob_simulated.get_identity()
        return state

    def _check_walkable(self, position: np.ndarray) -> bool:
        inbounds_horizontal = 0 < position[0] < self.env_settings.width
        inbounds_vertical = 0 < position[1] < self.env_settings.height

        if not inbounds_horizontal or not inbounds_vertical:
            return False

        for ob in self.objects.get_all():
            if np.all(ob.position == position) and not ob.walkable():
                return False
        return True

    def _find_object(self, position: np.ndarray) -> objects.GridObject | None:
        for ob in self.objects.get_non_wall():
            if np.all(ob.position == position):
                return ob
        return None

    def _get_info(self) -> dict[str, Any]:
        return {"Current Steps": self.step_count}

    def _get_obs(self) -> np.ndarray:
        return self.agent_pov.transform_obs(self.get_state(), self.objects.agent)

    def _render_frame(self) -> np.ndarray | None:
        # Define Canvas for new Frame
        window_size = (self.render_settings.window_width, self.render_settings.window_height)
        canvas = pygame.Surface(window_size)
        canvas.fill((255, 255, 255))

        # Draw Objects
        tilesize = window_size[0] / self.env_settings.width
        for ob in self.objects.get_all():
            ob.render(canvas, tilesize)

        # Draw Grid Lines
        line_color = (210, 210, 210)
        for x in range(self.env_settings.height + 1):
            pygame.draw.line(
                canvas,
                line_color,
                (0, tilesize * x),
                (window_size[0], tilesize * x),
                width=3,
            )
        for y in range(self.env_settings.width + 1):  
            pygame.draw.line(
                canvas,
                line_color,
                (tilesize * y, 0),
                (tilesize * y, window_size[1]),
                width=3,
            )

        # Display Canvas in Window
        if self.render_settings.render_mode == "human":
            assert self.render_settings.window is not None, "No window defined"
            assert self.render_settings.clock is not None, "No clock defined"
            self.render_settings.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.render_settings.clock.tick(self.render_settings.render_fps)

        # Return for rgb_array
        elif self.render_settings.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
        return None
