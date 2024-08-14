from typing import override

import numpy as np

from core import objects
from core.agentpov import AgentPOV
from core.gridengine import GridEngine
from utils.constants import MAP_DISTRACTIVE
from utils.dataclasses import EnvironmentSettings, RenderSettings, EnvironmentObjects


class DistractiveEnv(GridEngine):
    """Placeholder"""

    @override
    def __init__(
        self,
        agentPOV: AgentPOV | str = "global",
        render_mode: str | None = None,
        window_width: int = 1200
        ) -> None:

        env_settings = EnvironmentSettings(
            min_steps = 40,
            max_steps = 50,
            width = 23,
            height = 7,
            reward_range = (0,1),
        )

        render_settings = RenderSettings(
            render_mode = render_mode,
            window_width = window_width,
            window_height = int(window_width * (env_settings.height / env_settings.width)),
        )

        other_objects = np.array([
            objects.SmallReward((8,5), reward = 0.1, color=8),
            objects.SmallReward((6,1), reward = 0.1, color=8),
            objects.SmallReward((4,5), reward = 0.1, color=8),
            objects.SmallReward((2,1), reward = 0.1, color=8),
            objects.SmallReward((1,5), reward = 0.1, color=8),
        ])

        env_objects = EnvironmentObjects(
            agent = objects.Agent((11,1), color=1, state=3),
            target = objects.Target((21,5), color=2),
            walls = self.load_walls(MAP_DISTRACTIVE),
            other = other_objects,
        )

        super().__init__(
            env_settings=env_settings,
            render_settings=render_settings,
            env_objects=env_objects,
            agent_pov=agentPOV,
        )

    def _task(self) -> bool:
        return bool(np.all(self.objects.target.position == self.objects.agent.position))
