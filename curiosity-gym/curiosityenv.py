from gridenv import GridEnv
import objects as ob
import numpy as np
from agentpov import TwoFrontView
from constants import TWO_ROOMS_MAP
from agentpov import AgentPOV
from typing import override

class CuriousNavigation(GridEnv):
    
    def __init__(self, agentPOV: AgentPOV, render_mode=None, window_width:int=512):
        super().__init__(
            agentPOV, 
            render_mode=render_mode, 
            window_width=window_width,
            width=10,
            height=10,
            max_steps=50,
            )
    
    def _init_map(self) -> np.ndarray:
        return self.load_map(TWO_ROOMS_MAP)
    
    def _init_objects(self) -> np.ndarray:
        self.target = ob.Target((8,6))
        return np.array([
            self.target,
            ob.Door((3,4), state=2, color=1),
            ob.Key((3,1), color=1)
        ])
    
    def _init_agent(self) -> ob.GridObject:
        return ob.Agent((1,1), (1,0))

    def _get_reward(self) -> int:
        return 10/self.step_count if np.all(self.target.position == self.agent.position) else 0
    
    def _get_terminated(self) -> bool:
        if super()._get_terminated() or np.all(self.target.position == self.agent.position):
             return True
        return False
