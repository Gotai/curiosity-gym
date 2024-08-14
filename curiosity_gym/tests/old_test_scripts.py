import time
import unittest
from dataclasses import dataclass

import numpy as np

from core.gridengine import GridEngine


@dataclass
class Testcase(unittest.TestCase):
    identifier: int
    environment: GridEngine
    actions: list[int]
    position: tuple[int,int]
    state: int
    reward: float
    terminated: bool
    reward_max: float | None = None

    def test(self) -> bool:
        self.environment.reset()
        reward_total = 0
        for action in self.actions:
            _, reward, terminated, _, _ = self.environment.step(action)
            reward_total += reward

        agent = self.environment.objects.agent

        # Test agent position
        assert np.all(agent.position == self.position), (
            f"Unexpected agent position {agent.position} in test with id {self.identifier}."
            f" Expected {self.position}."
        )

        # Test agent state
        assert agent.state == self.state, (
            f"Unexpected agent state {agent.state} in test with id {self.identifier}."
            f" Expected {self.state}."
        )

        # Test final reward
        assert reward == self.reward, (
            f"Unexpected final reward {reward} in test with id {self.identifier}."
            f" Expected {self.reward}."
        )

        # Test termination status
        assert terminated == self.terminated, (
            f"Unexpected termination status {terminated} in test with id {self.identifier}."
            f" Expected {self.terminated}."
        )

        # Test maximum total reward for an episode
        assert self.reward_max is None or self.reward_max > reward_total, (
            f"Total reward {reward_total} is greater than expected maximum reward"
            f" of {self.reward_max} in test with id {self.identifier}."
        )

        return True


@dataclass
class Testsuite:
    cases: list[Testcase]

    def run_all(self):
        for case in self.cases:
            case.test()
        print("All test cases passed.")

    def run(self, identifier):
        for case in self.cases:
            if case.identifier == identifier:
                case.test()
                print(f"Test with id {identifier} passed.")

    def show(self, identifier):
        for case in self.cases:
            if case.identifier == identifier:
                case.environment.render_settings.render_mode = "human"
                case.environment.init_render()
                case.test()
                time.sleep(2)
                case.environment.render_settings.render_mode = None
                case.environment.close()
                print(f"Test with id {identifier} passed.")
