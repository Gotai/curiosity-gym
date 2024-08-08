from typing import override

import numpy as np

from core import objects
from core.agentpov import AgentPOV
from core.gridenv import GridEnv
from utils.constants import SPARSE_MAP_LARGE
from utils.dataclasses import EnvironmentSettings, RenderSettings, EnvironmentObjects


class SparseEnv(GridEnv):
    """Placeholder"""

    @override
    def __init__(self, agentPOV: AgentPOV, render_mode=None, window_width:int=512) -> None:

        env_settings = EnvironmentSettings(
            max_steps = 100,
            width = 15,
            height = 11,
            reward_range = (0,1),
        )

        render_settings = RenderSettings(
            render_mode = render_mode,
            window_width = window_width,
            window_height = int(window_width * (env_settings.height / env_settings.width)),
        )

        other_objects = np.array([
            objects.Door((9,2), state=2, color=3),
            objects.Key((5,2), color=3),
            objects.Door((12,4), state=2, color=4),
            objects.Key((13,1), color=4),
            objects.Door((8,6), state=2, color=5),
            objects.Key((11,8), color=5),
            objects.RandomBlock((6,6)),
            objects.Door((4,8), state=2, color=6),
            objects.Key((5,6), color=6),
            objects.Enemy((10,9), state=1, reach=4),
            objects.Enemy((1,5), state=0, reach=2),
        ])

        env_objects = EnvironmentObjects(
            agent = objects.Agent((1,1), color=1),
            target = objects.Target((7,4), color=2),
            walls = self.load_walls(SPARSE_MAP_LARGE),
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
