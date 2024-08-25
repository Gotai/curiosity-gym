Agent pov
---------
.. currentmodule:: curiosity_gym.core.agentpov

The point-of-view classes define the type of actions an RL agent can take and
the subset of the environment state the agent will receive from an environment 
as observation. In this way, the agent can be provided with a different set of
actions and information about the state of the environment. All pov classes 
inherit attributes and methods from the abstract base class :class:`AgentPOV`

Base class
~~~~~~~~~~
.. autosummary::
    :toctree: api/
    :nosignatures:

    AgentPOV

POV types
~~~~~~~~~
.. autosummary::
    :toctree: api/
    :nosignatures:
    
    GlobalView
    LocalView
    ForwardView
