Environments
------------
.. currentmodule:: curiosity_gym.core
The Curiosity Gym package provides multiple Gymnasium-based grid environments to test and benchmark
curiosity-driven RL algorithms. The dynamics of the grid are implemented in the abstract base class
:class:`~gridengine.GridEngine`. A concrete environment can be created by inheriting these dynamics and
providing custom :class:`~curiosity_gym.utils.dataclasses.EnvironmentSettings`, 
:class:`~curiosity_gym.utils.dataclasses.RenderSettings` and 
:class:`~curiosity_gym.utils.dataclasses.EnvironmentObjects` for the environment.


Grid engine
~~~~~~~~~~~
.. autosummary::
    :toctree: api/
    :nosignatures:

    ~gridengine.GridEngine


.. currentmodule:: curiosity_gym.envs

Environment types
~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: api/
    :nosignatures:
    
    ~distractiveenv.DistractiveEnv
    ~multitaskenv.MultitaskEnv
    ~sparseenv.SparseEnv
