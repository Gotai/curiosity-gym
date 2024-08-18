import unittest
from typing import override

import numpy as np

from curiosity_gym.core import objects
from curiosity_gym.envs.sparseenv import SparseEnv
from curiosity_gym.envs.distractiveenv import DistractiveEnv
from curiosity_gym.envs.multitaskenv import MultitaskEnv

env_sparse = SparseEnv()
env_distractive = DistractiveEnv()
env_multitask_1 = MultitaskEnv(task=1)
env_multitask_2 = MultitaskEnv(task=2)


class TestSparseEnv(unittest.TestCase):

    @override
    def setUp(self) -> None:
        env_sparse.reset()
        self.agent = env_sparse.objects.agent
        self.reward_max = env_sparse.env_settings.reward_range[1]

    def _assert_position(self, agent_pos: list[int], expected_pos: list[int]):
        self.assertListEqual(agent_pos, expected_pos,
                             f"Unexpected agent position {agent_pos}. "
                             f"Expected {expected_pos}.")

    def test_open_door_without_key(self) -> None:
        """Attempting to open closed door without key.

        Actions: Move to door, interact with door, attempt to move through.
        Expected: Agent is stuck before door, door remains closed.
        """
        for a in [0] * 7 + [1,0,2,3,0,0]:
            env_sparse.step(a)
        door = env_sparse.find_object(np.array([9,2]))
        assert isinstance(door, objects.Door), "Door not found at position [9,2] in SparseEnv."
        self._assert_position(self.agent.position.tolist(), [8,2])
        self.assertEqual(door.state, 2, f"Unexpected door state {door.state}. "
                         f"Expected it to be closed (2).")


    def test_open_door_with_key(self) -> None:
        """Open closed door with key. 

        Actions: Move to key, interact with key, move to door, interact with door, move through.
        Expectecd: Agent is able to move through, door is open.
        """
        for a in [0,0,0,0,1,3,0,2,0,0,0,3,0,0]:
            env_sparse.step(a)
        door = env_sparse.find_object(np.array([9,2]))
        assert isinstance(door, objects.Door), "Door not found at position [9,2] in SparseEnv."
        self._assert_position(self.agent.position.tolist(), [10,2])
        self.assertEqual(door.state, 0, f"Unexpected door state {door.state}. "
                         f"Expected it to be open (0).")

    def test_enemy_contact(self) -> None:
        """Termination by enemy contact.

        Actions: Move to enemy position.
        Expected: Episode termination.
        """
        enemy = env_sparse.find_object(np.array([10,9]))
        for a in [0,0,0,0,1,3,0,2,0,0,0,3,0,0,0,0,0,2,3,2,0,2,0,3,0,0,0,1,0,0,3,3,3,3,3]:
            _, _, terminated, _, _ = env_sparse.step(a)
        assert isinstance(enemy, objects.Enemy), "Enemy not found at position [10,9] in SparseEnv."
        self.assertListEqual(enemy.position.tolist(), self.agent.position.tolist(),
                             f"There is no enemy on agent position {self.agent.position}")
        self.assertTrue(terminated, "Termination by enemy contact failed.")

    def test_sparseenv_target(self) -> None:
        """Successful completion of the sparse reward environment.

        Actions: Optimal policy for standard sparse reward environment.
        Expected: Maximum reward, episode termination.
        """
        for a in [0,0,0,0,1,3,0,2,0,0,0,3,0,0,0,0,0,2,3,2,0,2,0,3,0,0,0,0,1,0,2,3,1,
                  1,3,0,2,0,0,3,0,0,2,0,1,0,0,1,3,1,1,0,1,3,0,0,1,0,0,0,0,1,0,0,0,0]:
            _, reward, terminated, _, _ = env_sparse.step(a)
        self.assertEqual(reward, self.reward_max,
                         f"Completion of sparse environment yielded reward of {reward}. "
                         f"Expected maximum reward of {self.reward_max}")
        self.assertTrue(terminated, "Termination by completing sparse environment failed.")


class TestDistractiveEnv(unittest.TestCase):

    @override
    def setUp(self) -> None:
        env_distractive.reset()
        self.agent = env_distractive.objects.agent
        self.reward_max = env_distractive.env_settings.reward_range[1]

    def test_small_reward(self) -> None:
        """Functionality of small reward object.

        Actions: Move to small reward.
        Expected: Receive a reward between 0 and the maximum, no episode termination.
        """
        small_reward = env_distractive.find_object(np.array([8,5]))
        for a in [1,0,0,2,0,0,0,0,1,0]:
            _, reward, terminated, _, _ = env_distractive.step(a)
        assert isinstance(small_reward, objects.SmallReward), (
            "Small reward was not found at position [8,5] in DistractiveEnv.")
        self.assertListEqual(small_reward.position.tolist(), [-1,-1],
                            "Small reward did not vanish after collection.")
        self.assertTrue(0 < reward < self.reward_max,
                      f"Small reward {reward} is not between 0 and max reward {self.reward_max}")
        self.assertFalse(terminated, "Small reward incorrectly terminated the episode.")

    def test_sum_of_small_rewards(self) -> None:
        """Sum of small rewards should be less than maximum reward.

        Actions: Move over all rewards (left side).
        Expected: A reward less than the maximum reward, no epsiode termination.
        """
        reward_total = 0
        for a in [1,0,0,2,0,0,0,0,1,0,0,1,0,0,0,0,2,0,0,2,0,0,0,0,1,0,0,1,0,0,0,0,2,0,0,2,0,0,0,0]:
            _, reward, terminated, _, _ = env_distractive.step(a)
            reward_total += reward
        self.assertLess(reward_total, self.reward_max,
                        f"Sum of small rewards {reward_total} is"
                        f"larger than or equal to maximum reward {self.reward_max}.")
        self.assertFalse(terminated, "Small reward incorrectly terminated the episode.")

    def test_distractiveenv_target(self) -> None:
        """Successful completion of the distractive reward environment.

        Actions: Move to target location (right side).
        Expected: Maximum reward, episode termination.
        """
        for a in [2,0,0,1,0,0,0,0,2,0,0,2,0,0,0,0,1,0,0,1,0,0,0,0,2,0,0,2,0,0,0,0,1,0,0,1,0,0,0,0]:
            _, reward, terminated, _, _ = env_distractive.step(a)
        self.assertEqual(reward, self.reward_max,
                         f"Completion of distractive environment yielded reward of {reward}. "
                         f"Expected maximum reward of {self.reward_max}")
        self.assertTrue(terminated, "Termination by completing distractive environment failed.")


class TestMultitaskEnv1(unittest.TestCase):

    @override
    def setUp(self) -> None:
        env_multitask_1.reset()
        self.agent = env_multitask_1.objects.agent
        self.reward_max = env_multitask_1.env_settings.reward_range[1]

    def test_multitask_1_while_active(self) -> None:
        """Successful completion of task 1 of the multitask environment.

        Actions: Get key, open door, move to target location (left side).
        Expected: Maximum reward, episode termination.
        """
        for a in [0,0,2,0,3,0,2,0,0,1,3,0,0,0,0]:
            _, reward, terminated, _, _ = env_multitask_1.step(a)
        self.assertEqual(reward, self.reward_max,
                         f"Completion of multitask 1 environment yielded reward of {reward}. "
                         f"Expected maximum reward of {self.reward_max}")
        self.assertTrue(terminated, "Termination by completing multitask 1 environment failed.")

    def test_multitask_2_while_inactive(self) -> None:
        """Completion of task 2 while task 1 is the active task.

        Actions: Push ball to target location (right side).
        Expected: No reward, no episode termination.
        """
        for a in [1,0,0,3,0,3,0,3]:
            _, reward, terminated, _, _ = env_multitask_1.step(a)
        self.assertEqual(reward, 0,
                            f"Completion of task 2 in multitask 1 yielded a reward of {reward}. "
                            f"Expected 0.")
        self.assertFalse(terminated,
                        "Completion of task 2 terminated the episode while task 1 was active.")


class TestMultitaskEnv2(unittest.TestCase):

    @override
    def setUp(self) -> None:
        env_multitask_2.reset()
        self.agent = env_multitask_2.objects.agent
        self.reward_max = env_multitask_2.env_settings.reward_range[1]

    def test_push_ball(self) -> None:
        """Test functionality of ball object.

        Actions: Move to ball, interact, attempt to move forward 2 cells.
        Expected: Movement to previous ball position.
        """
        ball = env_multitask_2.find_object(np.array([12,3]))
        for a in [1,0,0,3,0,0]:
            env_multitask_2.step(a)
        assert isinstance(ball, objects.Ball), "Ball not found at position [12,3] in MultitaskEnv."
        self.assertListEqual(ball.position.tolist(), [13,3],
                             f"Unexpected ball position after push {ball.position}. "
                             "Expected [13,3]")
        self.assertListEqual(self.agent.position.tolist(), ball.start_position.tolist(),
                            "Agent was not able to move to pushed ball start position.")

    def test_multitask_2_while_active(self):
        """Successful completion of task 2 of the multitask environment.

        Actions: Push ball to target location (right side).
        Expected: Maximum reward, episode termination.
        """
        ball = env_multitask_2.find_object(np.array([12,3]))
        target = env_multitask_2.ball_target
        for a in [1,0,0,3,0,3,0,3]:
            _, reward, terminated, _, _ = env_multitask_2.step(a)
        assert isinstance(ball, objects.Ball), "Ball not found at position [12,3] in MultitaskEnv."
        self.assertListEqual(ball.position.tolist(), target.position.tolist(),
                             f"Unexpected ball position after push {ball.position}. "
                             f"Expected {target.position.tolist()}")
        self.assertEqual(reward, self.reward_max,
                         f"Completion of multitask 2 environment yielded reward of {reward}. "
                         f"Expected maximum reward of {self.reward_max}")
        self.assertTrue(terminated, "Termination by completing multitask 2 environment failed.")

    def test_multitask_1_while_inactive(self):
        """Completion of task 1 while task 2 is the active task.

        Actions: Get key, open door, move to target location (left side).
        Expected: No reward, no epsiode termination.
        """
        for a in [0,0,2,0,3,0,2,0,0,1,3,0,0,0,0]:
            _, reward, terminated, _, _ = env_multitask_2.step(a)
        self.assertEqual(reward, 0,
                        f"Completion of task 2 in multitask 1 yielded a reward of {reward}. "
                        f"Expected 0.")
        self.assertFalse(terminated,
                        "Completion of task 1 terminated the episode while task 2 was active.")

if __name__== '__main__':
    unittest.main()
