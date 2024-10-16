"""Definition of the curiosity-gym distractive rewards environment."""

from typing import override

import numpy as np

from curiosity_gym.core import objects
from curiosity_gym.core.agentpov import AgentPOV
from curiosity_gym.core.gridengine import GridEngine
from curiosity_gym.utils.constants import MAP_DISTRACTIVE
from curiosity_gym.utils.dataclasses import (
    EnvironmentSettings,
    RenderSettings,
    EnvironmentObjects,
)


class DistractiveEnv(GridEngine):
    """Defines the structure of the curiosity-gym distractive rewards environment.\n
    It consists of two corridors. The left corridor contains small but frequent
    rewards, while the right corridor contains a larger but sparse reward. The
    environment is designed to test the agent's capability of escaping local reward
    optima (represented by the left corridor) to find a global sparse optimum
    (represented by the right corridor). The maximum step count is set to 50, so
    that the agent can not obtain rewards from both corridors within one episode.

    Parameters
    ----------
    agentPOV : :class:`~curiosity_gym.core.agentpov.AgentPOV` | str, optional
        Object or string defining the observations and action spaces of the RL agent.
        Valid string values are *'global'*, *'local_W'* and *'forward_L_W'*, where
        *W* and *L* are integers defining the width and length of the respective POV.
        By default :class:`~curiosity_gym.core.agentpov.GlobalView`.
    render_mode : str | None, optional
        Render mode in which the environment is run. If render mode is *human*, the
        environment will be rendered in PyGame. By default None.
    window_width : int, optional
        Horizontal size of the PyGame window in *human* render mode. By default 1200.


    .. figure:: ../../source/images/DistractiveEnv_optimal.gif
        :width: 600
        :align: center

        Example of a DistractiveEnv episode with an optimal policy.
    """

    @override
    def __init__(
        self,
        agentPOV: AgentPOV | str = "global",
        render_mode: str | None = None,
        window_width: int = 1200,
    ) -> None:

        env_settings = EnvironmentSettings(
            min_steps=40,
            max_steps=50,
            width=23,
            height=7,
            reward_range=(0, 1),
        )

        render_settings = RenderSettings(
            render_mode=render_mode,
            window_width=window_width,
            window_height=int(
                window_width * (env_settings.height / env_settings.width)
            ),
        )

        other_objects = np.array(
            [
                objects.SmallReward((8, 5), reward=0.1),
                objects.SmallReward((6, 1), reward=0.1),
                objects.SmallReward((4, 5), reward=0.1),
                objects.SmallReward((2, 1), reward=0.1),
                objects.SmallReward((1, 5), reward=0.1),
            ]
        )

        env_objects = EnvironmentObjects(
            agent=objects.Agent((11, 1), state=3),
            target=objects.Target((21, 5), color=2),
            walls=self.load_walls(MAP_DISTRACTIVE),
            other=other_objects,
        )

        super().__init__(
            env_settings=env_settings,
            render_settings=render_settings,
            env_objects=env_objects,
            agent_pov=agentPOV,
        )

    @override
    def check_task(self) -> bool:
        """Check whether the agent has reached the target at the end of the right corridor.

        Returns
        -------
        bool
            True if the agent is at the target position, False otherwise.
        """
        return bool(np.all(self.objects.target.position == self.objects.agent.position))
