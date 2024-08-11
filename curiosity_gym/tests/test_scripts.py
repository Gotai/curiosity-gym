import time
from dataclasses import dataclass

import numpy as np

from core.gridenv import GridEnv
#from tests.test_cases import CASES


@dataclass
class Testcase:
    identifier: int
    environment: GridEnv
    actions: list[int]
    position: tuple[int,int]
    state: int
    reward: float
    terminated: bool

    def test(self) -> bool:
        self.environment.reset()
        for action in self.actions:
            _, reward, terminated, _, _ = self.environment.step(action)

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

        return True


@dataclass
class Testsuite:
    cases: list[Testcase]

    def run_all(self):
        for case in self.cases:
            case.test()
        print("All test cases passed.")

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
