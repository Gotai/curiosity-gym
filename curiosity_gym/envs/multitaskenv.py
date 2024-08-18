from typing import override

import numpy as np

from curiosity_gym.core import objects
from curiosity_gym.core.agentpov import AgentPOV
from curiosity_gym.core.gridengine import GridEngine
from curiosity_gym.utils.constants import MAP_MULTITASK
from curiosity_gym.utils.dataclasses import EnvironmentSettings, RenderSettings, EnvironmentObjects


class MultitaskEnv(GridEngine):
    """Placeholder"""

    @override
    def __init__(
        self,
        agentPOV: AgentPOV | str = "global",
        task: int = 1,
        render_mode: str | None = None,
        window_width: int = 1200
        ) -> None:

        assert task <= 4, f"Invalid task id: {task}"
        self.task = task

        env_settings = EnvironmentSettings(
            min_steps = 15,
            max_steps = 50,
            width = 19,
            height = 7,
            reward_range = (0,1),
        )

        render_settings = RenderSettings(
            render_mode = render_mode,
            window_width = window_width,
            window_height = int(window_width * (env_settings.height / env_settings.width)),
        )

        self.ball_target = objects.Target((15,3), color=5)
        self.ball = objects.Ball((12,3), zone_low=(13,1), zone_high=(17,5), color=5)
        other_objects = np.array([
            objects.Door((6,3), color=3, state=2),
            objects.Key((7,1), color=3),
            self.ball_target,
            self.ball,
        ])

        env_objects = EnvironmentObjects(
            agent = objects.Agent((9,3), state=1),
            target = objects.Target((3,3), color=2),
            walls = self.load_walls(MAP_MULTITASK),
            other = other_objects,
        )

        super().__init__(
            env_settings=env_settings,
            render_settings=render_settings,
            env_objects=env_objects,
            agent_pov=agentPOV,
        )

    def _task(self) -> bool:
        if self.task % 2 == 1:
            return bool(np.all(self.objects.target.position == self.objects.agent.position))
        return bool(np.all(self.ball.position == self.ball_target.position))

    @override
    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        super().reset()
        if self.task > 2:
            self.objects.target.position = np.random.randint(1,6,2)
            self.ball_target.position = np.array([np.random.randint(14,18), np.random.randint(1,6)])

        if self.task % 2 == 1:
            pos = self.objects.target.position
            self.env_settings.min_steps = 18 - pos[0] + abs(3 - pos[1]) + (3 != pos[1])

        else:
            pos = self.ball_target.position
            self.env_settings.min_steps = 2 * pos[0] - 22 + (pos[1] != 3) * (2 * abs(pos[1]-3) + 5)

        return (self._get_obs(), self._get_info())
