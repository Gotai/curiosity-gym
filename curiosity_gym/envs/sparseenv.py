from typing import override

import numpy as np

from curiosity_gym.core import objects
from curiosity_gym.core.agentpov import AgentPOV
from curiosity_gym.core.gridengine import GridEngine
from curiosity_gym.utils.constants import MAP_SPARSE
from curiosity_gym.utils.dataclasses import EnvironmentSettings, RenderSettings, EnvironmentObjects


class SparseEnv(GridEngine):
    """Placeholder"""

    @override
    def __init__(
        self,
        agentPOV: AgentPOV | str = "global",
        render_mode: str | None = None,
        window_width: int = 800
        ) -> None:

        env_settings = EnvironmentSettings(
            min_steps = 66,
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
            # Room 1:
            objects.Key((5,2), color=3),
            objects.Door((9,2), state=2, color=3),

            # Room 2:
            objects.Key((13,1), color=4),
            objects.Door((12,4), state=2, color=4),

            # Room 3:
            objects.Key((11,8), color=5),
            objects.Door((8,6), state=2, color=5),        
            objects.Enemy((10,9), state=1, reach=4),

            # Room 4:
            objects.Key((5,6), color=6),
            objects.Door((4,8), state=2, color=6),
            objects.RandomBlock((6,6)),

            # Room 5:
            objects.Enemy((1,5), state=0, reach=2),
        ])

        env_objects = EnvironmentObjects(
            agent = objects.Agent((1,1)),
            target = objects.Target((7,4), color=2),
            walls = self.load_walls(MAP_SPARSE),
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
