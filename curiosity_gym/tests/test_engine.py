import unittest
from typing import override

from curiosity_gym.envs.multitaskenv import MultitaskEnv

env = MultitaskEnv(task=1)

class TestEngine(unittest.TestCase):

    @override
    def setUp(self) -> None:
        env.reset()
        self.agent = env.objects.agent

    def _assert_position(self, agent_pos: list[int], expected_pos: list[int]):
        self.assertListEqual(agent_pos, expected_pos,
                             f"Unexpected agent position {agent_pos}. "
                             f"Expected {expected_pos}.")

    def test_maximum_steps(self) -> None:
        """Maximum step count termination.
        
        Actions: Rotate right until maximum step count is reached.
        Expected: Episode termination.
        """
        for a in [1] * env.env_settings.max_steps:
            _, _, terminated, _, _ = env.step(a)
        self.assertTrue(terminated, "Termination by reaching maximum step count failed.")

    def test_grid_movement(self) -> None:
        """Movement dynamics.

        Actions: Move two cells forward, one down, two back.
        Expected: Agent position one down from starting position.
        """
        for a in [0,0,1,0,1,0,0]:
            env.step(a)
        self._assert_position(self.agent.position.tolist(), [10,3])
        self.assertEqual(self.agent.state, 3,
                         f"Unexpected agent state {self.agent.state}. Expected 2")

    def test_move_against_wall(self) -> None:
        """Moving against walls.

        Actions: Move towards left side wall.
        Expected: Agent stays at starting position.
        """
        for a in [0,1,0,0,0]:
            env.step(a)
        self._assert_position(self.agent.position.tolist(), [11,2])

    def test_simulation(self) -> None:
        """Simulating next action.

        Actions: Move Forward one cell.
        Expected: Simulate method yields same observation as step execution.
        """
        obs_simulate = env.simulate(0)
        obs_step, _, _, _, _ = env.step(0)
        self.assertListEqual(obs_simulate.tolist(), obs_step.tolist(),
                             "Simulation is not equal to actual observation.")
