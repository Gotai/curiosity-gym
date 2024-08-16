import unittest
from typing import override

import numpy as np

from core import objects
from envs.multitaskenv import MultitaskEnv

envs = {
    "global": MultitaskEnv(agentPOV="global"),
    "local_1": MultitaskEnv(agentPOV="local_1"),
    "local_3": MultitaskEnv(agentPOV="local_3"),
    "forward_2_1": MultitaskEnv(agentPOV="forward_2_1"),
    "forward_4_3": MultitaskEnv(agentPOV="forward_4_3"),
}

class TestPOVGlobal(unittest.TestCase):

    @override
    def setUp(self) -> None:
        self.env = envs["global"]
        self.env.reset()
        self.observation, _, _, _, _ = self.env.step(0)

    def test_observation_shape(self) -> None:
        """Test if observation is part of observation space specified by the environment.
        
        Expected: Observation is of shape (width * height, 3).
        """
        self.assertTrue(self.env.observation_space.contains(self.observation),
                        "Observation of pov 'global' is not consistend with "
                        f"given observation space {self.env.observation_space.shape}.")

    def test_observation_corners(self) -> None:
        """Test if cells in corners of map are visible in observation.

        Expected: Observable walls at corners of the multitask environment.
        """
        pos_max = np.array([self.env.env_settings.width-1, self.env.env_settings.height-1])
        ix_max = pos_max[0] + pos_max[1] * self.env.env_settings.width

        self.assertEqual(self.observation[0][0], 2,
                         "Position (0,0) is not visible in observation with pov 'global'.")

        self.assertEqual(self.observation[ix_max][0], 2,
                         f"Position ({pos_max[0]},{pos_max[1]}) is not visible in observation "
                         "with pov 'global'.")


class TestPOVLocal(unittest.TestCase):

    @override
    def setUp(self) -> None:
        self.env_1 = envs["local_1"]
        self.env_3 = envs["local_3"]
        self.env_1.reset()
        self.env_3.reset()


    def test_observation_shape_size_1(self) -> None:
        """Test if observation is part of observation space specified by the environment.
        
        Expected: Observation is of shape (9,3).
        """
        observation, _, _, _, _ = self.env_1.step(0)
        self.assertTrue(self.env_1.observation_space.contains(observation),
                        "Observation of pov 'local_1' is not consistend with "
                        f"given observation space {self.env_1.observation_space.shape}.")

    def test_observation_shape_size_3(self) -> None:
        """Test if observation is part of observation space specified by the environment.
        
        Expected: Observation is of shape (49,3).
        """
        observation, _, _, _, _ = self.env_3.step(0)
        self.assertTrue(self.env_3.observation_space.contains(observation),
                        "Observation of pov 'local_1' is not consistend with "
                        f"given observation space {self.env_3.observation_space.shape}.")

    def test_agent_index(self) -> None:
        """Test if agent is always at the expected index of the observation.

        Expected: Agent is at the center (radius, radius) of the observation.
        """
        observation, _, _, _, _ = self.env_1.step(0)
        self.assertEqual(observation[1 + 1 * 3][0], 1,
                         "Agent is not in center of observations for pov 'local_1'.")
        observation, _, _, _, _ = self.env_3.step(0)
        self.assertEqual(observation[3 + 3 * 7][0], 1,
                         "Agent is not in center of observations for pov 'local_3'.")


class TestPOVForward(unittest.TestCase):

    @override
    def setUp(self) -> None:
        self.env_2_1 = envs["forward_2_1"]
        self.env_4_3 = envs["forward_4_3"]
        self.env_2_1.reset()
        self.env_4_3.reset()

    def test_observation_shape_size_2_1(self) -> None:
        """Test if observation is part of observation space specified by the environment.
        
        Expected: Observation is of shape (3,3).
        """
        observation, _, _, _, _ = self.env_2_1.step(0)
        self.assertTrue(self.env_2_1.observation_space.contains(observation),
                        "Observation of pov 'local_1' is not consistend with "
                        f"given observation space {self.env_2_1.observation_space.shape}.")

    def test_observation_shape_size_4_3(self) -> None:
        """Test if observation is part of observation space specified by the environment.
        
        Expected: Observation is of shape (12,3).
        """
        observation, _, _, _, _ = self.env_4_3.step(0)
        self.assertTrue(self.env_4_3.observation_space.contains(observation),
                        "Observation of pov 'local_1' is not consistend with "
                        f"given observation space {self.env_4_3.observation_space.shape}.")

    def test_agent_index(self) -> None:
        """Test if agent is always at the expected index of the observation.

        Expected: Agent is at the middle of the first row [(pov_width-1)/2] of the observation.
        """
        observation, _, _, _, _ = self.env_2_1.step(0)
        self.assertEqual(observation[0][0], objects.Agent.id,
                         "Agent is not at expected index of observations for pov 'forward_2_1'.")
        observation, _, _, _, _ = self.env_4_3.step(0)
        self.assertEqual(observation[1][0], objects.Agent.id,
                         "Agent is not at expected index of observations for pov 'forward_4_3'.")

    def test_see_through_walls(self) -> None:
        """Test if agent can see through walls.

        Actions: Move to door (left side).
        Expected: Agent can not see target object through wall.
        """
        for a in [2,0,0]:
            observation, _, _, _, _ = self.env_4_3.step(a)
        self.assertNotEqual(observation[10][0], objects.Target.id,
                            "Agent can see target through wall in multitask environment.")

    def test_see_objects_relative_to_agent(self) -> None:
        """Test observation is constructed in relation to agent position and rotation.

        Actions: Turn right.
        Expected: Agent sees ball at index 10.
        """
        observation, _, _, _, _ = self.env_4_3.step(1)
        self.assertEqual(observation[10][0], objects.Ball.id,
                         "Agent can not see ball for pov 'forward_4_3' in multitask environment.")
