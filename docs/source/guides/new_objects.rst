=========================================
Creating new grid objects in CuriosityGym
=========================================

This guide explains how to create new grid object subclasses for the CuriosityGym framework. Each new subclass should inherit from the `GridObject` base class and implement its abstract methods to define custom behaviors.

Overview of :class:`~curiosity_gym.objects.GridObject`
--------------------------
The `GridObject` class is the base for all grid objects in CuriosityGym. It provides:

- **Attributes**: 
  - `position`, `color`, and `state` for defining grid object characteristics.
- **Methods**: 
  - `reset`, `step`, `simulate`, `interact`, `is_walkable`, and `is_harmful` for default behaviors.
- **Abstract Methods**: 
  - `render`, which must be implemented to define the visual representation of the object.

Steps to Create a New Grid Object Subclass
------------------------------------------

1. **Define a New Class**
   Define a new subclass inheriting from ``GridObject``.

   .. code-block:: python

      from curiosity_gym.objects import GridObject
      
      class CustomObject(GridObject):
          """Custom grid object with specific behavior."""

2. **Implement the Constructor**
   Define the ``__init__`` method to initialize attributes specific to your object. Call ``super().__init__`` to inherit and initialize common attributes (``position``, ``color``, ``state``).

   .. code-block:: python

      def __init__(self, position, color=0, state=0):
          super().__init__(position, color, state)
          # Add custom initialization logic here

3. **Override the ``render`` Method**
   Implement the ``render`` method to define how the object is visually represented in the environment. Use PyGame functions for drawing.

   .. code-block:: python

      def render(self, canvas, pixelsquare):
          pygame.draw.rect(
              canvas,
              IX_TO_COLOR[self.color],
              pygame.Rect(
                  pygame.Vector2(*(pixelsquare * self.position)),
                  (pixelsquare, pixelsquare)
              )
          )

4. **Customize Interaction (Optional)**
   Override the ``interact`` method to define how agents interact with your object.

   .. code-block:: python

      def interact(self, agent):
          """Define interaction logic with the agent."""
          # Custom interaction logic here

5. **Modify Behavior During Environment Steps (Optional)**
   Override the ``step`` method to define how the object behaves during a timestep.

   .. code-block:: python

      def step(self, action, front_object=None, walkable=False):
          """Custom behavior on each environment step."""
          # Custom step logic here
          return 0  # Optional reward

6. **Set Walkability (Optional)**
   Override the ``is_walkable`` method if the object allows agents to move over it.

   .. code-block:: python

      def is_walkable(self):
          return True  # Example: The object is walkable

7. **Define Harmful Objects (Optional)**
   Override ``is_harmful`` if the object should terminate the episode when the agent interacts with it.

   .. code-block:: python

      def is_harmful(self):
          return True  # Example: The object is harmful

8. **Register the Object (Automatic)**
   Each subclass is automatically registered with a unique identifier upon definition. No additional registration is required.

Example: Creating a "Bonus" Grid Object
---------------------------------------

The following example creates a ``Bonus`` object that rewards the agent upon interaction:

.. code-block:: python

   from curiosity_gym.objects import GridObject

   class Bonus(GridObject):
       """Grid object that rewards the agent when collected."""

       def __init__(self, position, reward=10):
           super().__init__(position, color=5, state=0)
           self.reward = reward

       def render(self, canvas, pixelsquare):
           pygame.draw.circle(
               canvas,
               IX_TO_COLOR[self.color],
               pygame.Vector2(*((self.position + [0.5, 0.5]) * pixelsquare)),
               0.3 * pixelsquare,
           )

       def step(self, action, front_object=None, walkable=False):
           return 0  # No specific behavior during timestep

       def interact(self, agent):
           """Collect the bonus and remove it from the grid."""
           self.position = np.array([-1, -1])  # Remove from grid
           return self.reward