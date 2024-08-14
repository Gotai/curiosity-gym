from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
import numpy as np
import pygame

from core import objects
from core.agentpov import AgentPOV, GlobalView, LocalView
from utils.enums import Action
from utils.dataclasses import EnvironmentSettings, RenderSettings, EnvironmentObjects


class GridEngine(gym.Env, ABC):
    """Placeholder"""

    # pylint: disable=too-many-instance-attributes
    # Additional instance attributes are required for this class, to cover the gym.Env interface.

    def __init__(
        self,
        env_settings: EnvironmentSettings,
        render_settings: RenderSettings,
        env_objects: EnvironmentObjects,
        agent_pov: AgentPOV | str,
    ) -> None:

        # Store settings
        self.env_settings = env_settings
        self.render_settings = render_settings
        self.reward_range = env_settings.reward_range

        # Initialise agent pov
        self.agent_pov = self._init_pov(agent_pov)
        self.action_space = self.agent_pov.action_space
        self.observation_space = self.agent_pov.observation_space

        # Current environment state
        self.objects = env_objects
        self.step_count = 0

        # Initialise render objects
        if self.render_settings.render_mode == "human":
            self.init_render()

    @abstractmethod
    def _task(self) -> bool:
        pass

    def step(self, action: int | Action) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        reward = 0
        for ob in self.objects.get_non_wall():
            reward += ob.step(
                Action(action),
                self.find_object(self.objects.agent.get_front()),
                self._check_walkable(self.objects.agent.get_front()),
                )

        self.step_count += 1
        reward += (self.env_settings.min_steps / self.step_count *
                   self.env_settings.reward_range[1]) if self._task() else 0
        obs = self._get_obs()

        if self.render_settings.render_mode == "human":
            pygame.display.set_caption(f"Curiosity Gym [Current Steps: {self.step_count}, " +
                                        f"Step reward: {reward}]")
            self._render_frame()

        return obs, reward, self._get_terminated(), False, self._get_info()

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
        state = np.zeros([self.env_settings.width * self.env_settings.height, 3], dtype=int)
        for ob in self.objects.get_all():
            x,y = ob.position
            assert x + y * self.env_settings.width < len(state), \
            f"""Position [{x},{y}] of object with type {self.get_object_ids()[ob.id]}
            is invalid for grid with size ({self.env_settings.width}, {self.env_settings.height})"""
            state[x + y * self.env_settings.width] = ob.get_identity()
        return state

    def init_render(self) -> None:
        pygame.init()
        pygame.display.init()
        window_size = (self.render_settings.window_width, self.render_settings.window_height)
        self.render_settings.window = pygame.display.set_mode(window_size)
        self.render_settings.clock = pygame.time.Clock()

    def load_walls(self, positions: np.ndarray) -> np.ndarray:
        walls = [objects.Wall(position) for position in positions]
        return np.array(walls)

    def simulate(self, action: int | Action) -> np.ndarray:
        state = np.zeros([self.env_settings.width * self.env_settings.height, 3])
        for ob in self.objects.get_all():
            ob_simulated = ob.simulate(
                Action(action),
                self.find_object(self.objects.agent.get_front()),
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

    def _check_harmful(self, position: np.ndarray) -> bool:
        for ob in self.objects.other:
            if np.all(ob.position == position) and ob.isHarmful():
                return True
        return False

    def find_object(self, position: np.ndarray) -> objects.GridObject | None:
        for ob in self.objects.get_non_wall():
            if np.all(ob.position == position):
                return ob
        return None

    def _get_info(self) -> dict[str, Any]:
        return {"Current Steps": self.step_count}

    def _get_obs(self) -> np.ndarray:
        return self.agent_pov.transform_obs(self.get_state(), self.objects.agent)

    def _get_terminated(self) -> bool:
        return (self.step_count >= self.env_settings.max_steps
        or self._check_harmful(self.objects.agent.position)
        or self._task())

    def _init_pov(self, agent_pov: AgentPOV | str) -> AgentPOV:
        assert isinstance(agent_pov, (AgentPOV, str)), (
            f"Invalid type of agent_pov: {type(agent_pov)}")

        if isinstance(agent_pov, AgentPOV):
            return agent_pov

        if agent_pov.lower() == "global":
            return GlobalView(self.env_settings.width, self.env_settings.height)

        if agent_pov.lower().startswith("local_"):
            radius = agent_pov[6:]
            assert radius.isnumeric(), f"Invalid radius for local pov: {radius}"
            return LocalView(int(radius), self.env_settings.width, self.env_settings.height)

        raise ValueError(f"Invalid agent pov: {agent_pov}.")

    def _render_frame(self) -> np.ndarray | None:
        # Define canvas for new Frame
        window_size = (self.render_settings.window_width, self.render_settings.window_height)
        canvas = pygame.Surface(window_size)
        canvas.fill((255, 255, 255))

        # Draw objects
        tilesize = window_size[0] / self.env_settings.width
        for ob in self.objects.get_all():
            ob.render(canvas, tilesize)

        # Draw grid lines
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

        # Add overlay for visible cells
        for pos in self.agent_pov.visible_positions:
            overlay = pygame.Surface((tilesize, tilesize))
            overlay.set_alpha(30)
            overlay.fill((255,153,20))
            canvas.blit(overlay, (pos[0] * tilesize, pos[1] * tilesize))

        # Display canvas in window
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
