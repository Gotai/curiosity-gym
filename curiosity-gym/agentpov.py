from abc import ABC, abstractmethod
import numpy as np
from gymnasium import spaces

class AgentPOV(ABC):

    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
       
    @abstractmethod
    def transform_obs(self, state, agent):
        pass
    
    def transform_action(self, action):
        return action

    def _position_to_ix(self, position, size):
        x,y = position
        return y*size + x



class TwoFrontView(AgentPOV):
    
    def __init__(self):
        action_space = spaces.Discrete(4)
        observation_space = spaces.Box(0, np.array([[7,7,3], [7,7,3]]), (2,3), dtype=np.float64)
        super().__init__(action_space, observation_space)
    
    def transform_obs(self, state, agent):
        assert(agent), f"There is no agent defined for this environment."
        size = int(state.shape[0]**0.5)
        ix1 = self._position_to_ix(agent.position + agent.rotation * np.array([1,-1]), size)
        ix2 = self._position_to_ix(agent.position + agent.rotation*2 * np.array([1,-1]), size)
        return np.array([state[ix1], state[ix2]])
    
    def transform_action(self, action):
        return action
    
class OneFrontView(AgentPOV):

    def __init__(self):
        action_space = spaces.Discrete(4)
        observation_space = spaces.Box(0, 7, (3,), dtype=np.float64)
        super().__init__(action_space, observation_space)
    
    def transform_obs(self, state, agent):
        assert(agent), f"There is no agent defined for this environment."
        size = int(state.shape[0]**0.5)
        ix1 = self._position_to_ix(agent.position + agent.rotation * np.array([1,-1]), size)
        return state[ix1]
    

class FullView(AgentPOV):

    def __init__(self):
        action_space = spaces.Discrete(4)
        observation_space = spaces.Box(0, 7, (100,3), dtype=np.float64)
        super().__init__(action_space, observation_space)

    def transform_obs(self, state, agent):
        return state