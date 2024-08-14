from core.agentpov import FullView
from envs.sparseenv import SparseEnv
from envs.distractiveenv import DistractiveEnv
from envs.multitaskenv import MultitaskEnv
from tests.old_test_scripts import Testcase, Testsuite


env_sparse = SparseEnv(FullView())
env_distractive = DistractiveEnv(FullView())
env_multitask_1 = MultitaskEnv(FullView(), task=1)
env_multitask_2 = MultitaskEnv(FullView(), task=2)

testsuite = Testsuite([
    # Case 1: Maximum step count.
    #
    # Objective: Test step counting and episode termination.
    # Actions: Rotate right until maximum step count is reached.
    # Expected: Episode termination.
    Testcase(
        identifier = 1,
        environment = env_sparse,
        position = (1,1),
        state = 0,
        reward = 0,
        terminated = True,
        actions = [1] * env_sparse.env_settings.max_steps,
    ),

    # Case 2: Movement dynamics.
    #
    # Objective: Test whether movement in grid behaves as expected.
    # Actions: Move two cells forward, one down, two back.
    # Expected: Agent position one down from starting position.
    Testcase(
        identifier = 2,
        environment = env_sparse,
        position = (1,2),
        state = 2,
        reward = 0,
        terminated = False,
        actions = [0,0,1,0,1,0,0],
    ),

    # Case 3: Moving against walls.
    #
    # Objective: Test whether walls stop agent movement.
    # Actions: Move towards left side wall.
    # Expected: Agent stays at starting position.
    Testcase(
        identifier = 3,
        environment = env_sparse,
        position = (1,2),
        state = 2,
        reward = 0,
        terminated = False,
        actions = [0,0,1,0,1,0,0],
    ),

    # Case 4: Open closed door with key.
    #
    # Objective: Test whether key and door work as expected.
    # Actions: Move to key, pick up key, move to door, open door, move through.
    # Expected: Agent is able to move through door.
    Testcase(
        identifier = 4,
        environment = env_sparse,
        position = (10,2),
        state = 0,
        reward = 0,
        terminated = False,
        actions = [0,0,0,0,1,3,0,2,0,0,0,3,0,0],
    ),

    # Case 5: Attempt to open door without key.
    #
    # Objective: Test whether door stays closed after attempting to open it without key.
    # Actions: Move to door, attempt to open, attempt to move through.
    # Expected: Agent is stuck at position in front of door.
    Testcase(
        identifier = 5,
        environment = env_sparse,
        position = (8,2),
        state = 0,
        reward = 0,
        terminated = False,
        actions = [0] * 7 + [1,0,2,3,0,0],
    ),

    # Case 6: Enemy contact.
    #
    # Objective: Test interaction between agent and enemy positions.
    # Actions: Move to enemy position.
    # Expected: Episode termination.
    Testcase(
        identifier = 6,
        environment = env_sparse,
        position = (10,6),
        state = 2,
        reward = 0,
        terminated = True,
        actions = [0,0,0,0,1,3,0,2,0,0,0,3,0,0,0,0,0,2,3,2,0,2,0,3,0,0,0,1,0,0,3,3,3,3,3],
    ),

    # Case 7: Target sparse reward.
    #
    # Objective: Test the successful completion of the task.
    # Actions: Optimal policy for sparse navigation environment.
    # Expected: Maximum reward and episode termination.
    Testcase(
        identifier = 7,
        environment = env_sparse,
        position = (7,4),
        state = 0,
        reward = env_sparse.env_settings.reward_range[1],
        terminated = True,
        actions = [0,0,0,0,1,3,0,2,0,0,0,3,0,0,0,0,0,2,3,2,0,2,0,3,0,0,0,0,1,0,2,3,1,1,3,0,
                   2,0,0,3,0,0,2,0,1,0,0,1,3,1,1,0,1,3,0,0,1,0,0,0,0,1,0,0,0,0],
    ),

    # Case 8: Small reward.
    #
    # Objective: Test the functionality of the small reward object.
    # Actions: Move to small reward.
    # Expected: A reward != 0 without episode termination.
    Testcase(
        identifier = 8,
        environment= env_distractive,
        position = (8,5),
        state = 2,
        reward = 0.1,
        terminated = False,
        actions = [1,0,0,2,0,0,0,0,1,0],
    ),

    # Case 9: Sum of small rewards.
    #
    # Objective: Test distractive characteristic of small rewards.
    # Actions: Move over all small rewards.
    # Expected: A reward less then the maxium reward without episode termination.
    Testcase(
        identifier = 9,
        environment= env_distractive,
        position = (1,5),
        state = 3,
        reward = 0.1,
        terminated = False,
        actions = [1,0,0,2,0,0,0,0,1,0,0,1,0,0,0,0,2,0,0,2,0,0,0,0,1,0,0,1,0,0,0,0,2,0,0,2,0,0,0,0],
        reward_max = env_distractive.env_settings.reward_range[1]
    ),

    # Case 9: Target distractive reward.
    #
    # Objective: Test completion of distractive reward environment.
    # Actions: Move to target location (right side).
    # Expected: Maximum reward + episode termination.
    Testcase(
        identifier = 10,
        environment= env_distractive,
        position = (21,5),
        state = 3,
        reward = env_distractive.env_settings.reward_range[1],
        terminated = True,
        actions = [2,0,0,1,0,0,0,0,2,0,0,2,0,0,0,0,1,0,0,1,0,0,0,0,2,0,0,2,0,0,0,0,1,0,0,1,0,0,0,0],
    ),

    # Case 10: Pushing ball.
    #
    # Objective: Test if ball is pushed one cell by interaction.
    # Actions: Move to ball, interact, attempt to move forward 2 cells.
    # Expected: Movement to previous ball position possible.
    Testcase(
        identifier = 10,
        environment= env_multitask_1,
        position = (12,3),
        state = 0,
        reward = 0,
        terminated = False,
        actions = [1,0,0,3,0,0],
    ),

    # Case 11: Task 1 of multitask.
    #
    # Objective: Test completion of task 1 in the multitask environment.
    # Actions: Get key, open the door, move to target location (left side).
    # Expected: Maximum reward + episode termination.
    Testcase(
        identifier = 11,
        environment= env_multitask_1,
        position = (3,3),
        state = 2,
        reward = env_distractive.env_settings.reward_range[1],
        terminated = True,
        actions = [0,0,2,0,3,0,2,0,0,1,3,0,0,0,0],
    ),

    # Case 12: Task 1 of multitask: No reward for task 2.
    #
    # Objective: Test completion of task 2 while task 1 is active.
    # Actions: Push ball to target location (right side).
    # Expected: No reward + no episode termination.
    Testcase(
        identifier = 12,
        environment= env_multitask_1,
        position = (13,3),
        state = 0,
        reward = 0,
        terminated = False,
        actions = [1,0,0,3,0,3,0,3],
    ),

    # Case 13: Task 2 of multitask.
    #
    # Objective: Test completion of task 2 in the multitask environment.
    # Actions: Push ball to target location (right side).
    # Expected: Maximum reward + episode termination.
    Testcase(
        identifier = 13,
        environment= env_multitask_2,
        position = (13,3),
        state = 0,
        reward = env_distractive.env_settings.reward_range[1],
        terminated = True,
        actions = [1,0,0,3,0,3,0,3],
    ),

    # Case 14: Task 2 of multitask: No reward for task 1.
    #
    # Objective: Test completion of task 1 while task 2 is active.
    # Actions: Get key, open the door, move to target location (left side).
    # Expected: Maximum reward + episode termination.
    Testcase(
        identifier = 14,
        environment= env_multitask_2,
        position = (3,3),
        state = 2,
        reward = 0,
        terminated = False,
        actions = [0,0,2,0,3,0,2,0,0,1,3,0,0,0,0],
    ),
])
