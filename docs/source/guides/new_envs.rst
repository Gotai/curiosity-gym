=========================================
Creating new environments in CuriosityGym
=========================================

This guide explains how to create custom environments for the CuriosityGym framework. Each custom environment should inherit from the `GridEngine` base class and implement its required methods to define unique behavior.

Overview of ``GridEngine``
--------------------------
The `GridEngine` class is the abstract base class for all environments in CuriosityGym. It provides:

- **Attributes**:
  - `env_settings`: Defines environment dimensions, step limits, and reward range.
  - `render_settings`: Controls the rendering mode and display configurations.
  - `env_objects`: Stores all objects within the environment (e.g., agent, walls, rewards).
- **Methods**:
  - `step`, `reset`, `render`, and `close` to support the Gymnasium API.
  - `check_task`, which must be implemented by subclasses to define task completion criteria.
- **Utilities**:
  - Functions for simulating actions, rendering, and managing grid objects.

Steps to Create a New Custom Environment
-----------------------------------------

1. **Define a Subclass**
   Create a new class that inherits from ``GridEngine``.

   .. code-block:: python

      from curiosity_gym.core.gridengine import GridEngine
      
      class CustomEnv(GridEngine):
          """Custom environment with unique grid dynamics."""

2. **Initialize the Environment**
   Implement the ``__init__`` method to define the grid structure, objects, and settings. Call the base class constructor using ``super().__init__``.

   .. code-block:: python

      def __init__(self, agentPOV="global", render_mode=None, window_width=1200):
          # Define environment settings
          env_settings = EnvironmentSettings(
              min_steps=10,
              max_steps=100,
              width=10,
              height=10,
              reward_range=(0, 1),
          )

          # Define render settings
          render_settings = RenderSettings(
              render_mode=render_mode,
              window_width=window_width,
              window_height=int(window_width * (env_settings.height / env_settings.width)),
          )

          # Define environment objects
          env_objects = EnvironmentObjects(
              agent=objects.Agent((1, 1), state=0),
              target=objects.Target((8, 8), color=2),
              walls=self.load_walls(np.array([(0, y) for y in range(10)])),  # Example walls
              other=np.array([objects.SmallReward((3, 3), reward=0.1)]),  # Example rewards
          )

          # Initialize base class
          super().__init__(
              env_settings=env_settings,
              render_settings=render_settings,
              env_objects=env_objects,
              agent_pov=agentPOV,
          )

3. **Define the Task**
   Implement the ``check_task`` method to define when the task is completed. This determines episode termination.

   .. code-block:: python

      def check_task(self) -> bool:
          """Task completion condition."""
          return bool(
              np.all(self.objects.agent.position == self.objects.target.position)
          )

4. **Test the Environment**
   Ensure your environment works with RL algorithms by testing it using the Gymnasium API:

   .. code-block:: python

      from curiosity_gym.environments import CustomEnv

      env = CustomEnv(render_mode="human")
      observation = env.reset()
      
      for _ in range(100):
          action = env.action_space.sample()  # Random action
          observation, reward, terminated, truncated, info = env.step(action)
          if terminated or truncated:
              break
      env.close()

Example: Creating a "Distractive Rewards" Environment
------------------------------------------------------
The following example creates an environment with two corridors: one with small frequent rewards and another with a larger sparse reward. The agent's task is to find the global reward optimum.

.. code-block:: python

   from curiosity_gym.core.gridengine import GridEngine
   from curiosity_gym.utils.dataclasses import EnvironmentSettings, RenderSettings, EnvironmentObjects
   from curiosity_gym.core import objects
   import numpy as np

   class DistractiveEnv(GridEngine):
       """Environment with distractive rewards."""

       def __init__(self, agentPOV="global", render_mode=None, window_width=1200):
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

           other_objects = np.array([
               objects.SmallReward((8, 5), reward=0.1),
               objects.SmallReward((6, 1), reward=0.1),
               objects.SmallReward((4, 5), reward=0.1),
               objects.SmallReward((2, 1), reward=0.1),
               objects.SmallReward((1, 5), reward=0.1),
           ])

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

       def check_task(self) -> bool:
           """Check if agent has reached the target."""
           return bool(np.all(self.objects.target.position == self.objects.agent.position))
