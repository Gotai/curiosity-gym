from typing import override

import numpy as np

import objects
from agentpov import AgentPOV
from constants import TWO_ROOMS_MAP
from gridenv import GridEnv
from utils import EnvironmentSettings, RenderSettings, EnvironmentObjects


class SmallNavigationEnv(GridEnv):
    """Placeholder"""

    @override
    def __init__(self, agentPOV: AgentPOV, render_mode=None, window_width:int=512) -> None:

        env_settings = EnvironmentSettings(
            max_steps = 50,
            width = 10,
            height = 10,
            reward_range = (0,1),
        )

        render_settings = RenderSettings(
            render_mode = render_mode,
            window_width = window_width,
            window_height = int(window_width * (env_settings.height / env_settings.width)),
        )

        other_objects = np.array([
            objects.Door((3,4), state=2, color=3),
            objects.Key((3,1), color=3),
            objects.RandomBlock((1,3)),
            objects.Enemy((6,7), state=1),
        ])

        env_objects = EnvironmentObjects(
            agent = objects.Agent((1,1), color=1),
            target = objects.Target((8,6), color=2),
            walls = self.load_walls(TWO_ROOMS_MAP),
            other = other_objects,
        )

        super().__init__(
            agentPOV,
            env_settings=env_settings,
            render_settings=render_settings,
            env_objects=env_objects,
        )

    @override
    def _get_reward(self) -> float:
        return 16 / self.step_count * self.env_settings.reward_range[1] if self._task() else 0

    @override
    def _get_terminated(self) -> bool:
        return super()._get_terminated() or self._task()

    def _task(self) -> bool:
        return bool(np.all(self.objects.target.position == self.objects.agent.position))
