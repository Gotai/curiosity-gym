Grid objects
------------
.. currentmodule:: curiosity_gym.core.objects

All elements that can be placed in a Curiosity-Gym environment are referred to as grid objects. 
They all inherit attributes and methods from the abstract base class :class:`.GridObject`.

Base class
~~~~~~~~~~
.. autosummary::
    :toctree: api/
    :nosignatures:

    GridObject

Object types
~~~~~~~~~~~~
.. autosummary::
    :toctree: api/
    :nosignatures:

    Agent
    Wall
    Target
    Door
    Key
    RandomBlock
    Enemy
    SmallReward
    Ball
