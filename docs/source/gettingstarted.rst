.. _getting-started:

===============
Getting started
===============

.. _install:

Installation
------------
Curiosity Gym requires **Python 3.12** or greater. It can be installed via pip after cloning the git repository::

    git clone https://github.com/chrisreimann/curiosity-gym
    cd curiosity-gym
    pip install .


Using predefined environments
-----------------------------
To use the predefined environments from the Curiosity Gym framework, simply import `SparseEnv`, `DistractiveEnv` or `MultitaskEnv` from the package. For Example::
    
    # Import predefined environment
    from curiosity_gym import SparseEnv

    # Initialize environment
    env = SparseEnv(agentPOV="local_2", render_mode="human")

    # Use Gymnasium API
    observation, info = env.reset()
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    env.close()

After an environment is initialized, all Gymnasium API methods, like `step`, `reset` or `close` can be used.
