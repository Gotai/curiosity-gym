from core.agentpov import FullView
from envs.sparseenv import SparseEnv
from tests.test_scripts import Testcase


env_sparse = SparseEnv(FullView())

"""Case 1: Maximum step count.

Objective: Test whether reaching the maximum step count terminates the episode.
Actions: Rotate right until maximum step count is reached.
"""
case_1 = Testcase(
    identifier = 1,
    environment = env_sparse,
    position = (1,1),
    state = 0,
    reward = 0,
    terminated = True,
    actions = [1]*env_sparse.env_settings.max_steps,
)
