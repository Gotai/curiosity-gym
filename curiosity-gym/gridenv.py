from abc import ABC, abstractmethod
import gymnasium as gym
import numpy as np
import pygame
from constants import OBJECT_TO_IDX
from agentpov import AgentPOV
from objects import GridObject
import objects as ob
from utils import EnvironmentSettings

class GridEnv(gym.Env, ABC):
    """Abstract base class for 2D grid world gymnasium environments.

    :param gym: _description_
    :type gym: _type_
    :param ABC: _description_
    :type ABC: _type_
    :return: _description_
    :rtype: _type_
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        agent_pov: AgentPOV,
        env_settings: EnvironmentSettings,
        render_mode: str | None = None,
        window_width: int = 512,
    ):

        # Get Action & Observations spaces
        self.agent_pov = agent_pov
        self.action_space = agent_pov.action_space
        self.observation_space = agent_pov.observation_space

        # Configure Env
        self.max_steps = env_settings.max_steps
        self.reward_range = env_settings.reward_range
        self.width = env_settings.width
        self.height = env_settings.heigth

        self.map_layout = self._init_map()
        self.objects = self._init_objects()
        self.agent = self._init_agent()

        # Configure Rendering
        self.render_mode = render_mode
        self.window_width = window_width
        self.window_height = window_width * (self.height / self.width)
        self.tile_size = window_width / self.width
        self.window = None
        self.clock = None


    @abstractmethod
    def _init_objects(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def _init_map(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def _init_agent(self) -> GridObject:
        pass
    
    @abstractmethod
    def _get_reward(self) -> int:
        pass
    
    @abstractmethod
    def _get_terminated(self) -> bool:
        if self.step_count >= self.max_steps:
            return True
    
    def reset(self, seed: int | None=None, options = None):
        super().reset(seed=seed)
        self.step_count = 0
        for object in self.objects:
            object.reset()
        self.agent.reset()
        return (self._get_obs(), self._get_info())

    def _get_obs(self):
        return self.agent_pov.transform_obs(self.get_state(), self.agent)
    
    def _get_info(self):
        return {}
    
    def step(self, action):
        self.step_count += 1

        front = self.agent.position + self.agent.rotation * np.array([1,-1])
        if action == 0 and self.check_walkable(front):
            self.agent.position = front

        if action == 1:
            self.agent.rotate(np.array([1,-1]))

        if action == 2:
            self.agent.rotate(np.array([-1,1]))

        if action == 3:
            object = self.find_object(front)
            if object:
                object.interact(self.agent)

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_reward(), self._get_terminated(), False, self._get_info()
    

    def check_walkable(self, position) -> bool:
        if position[0] >= self.width or position[1] >= self.height:
            return False
        
        if position[0] < 0 or position[1] < 0:
            return False
        
        for wall in self.map_layout:
            if np.all(wall.position == position):
                return False
            
        for object in self.objects:
            if np.all(object.position == position) and not object.walkable():
                return False
        return True
    
    def find_object(self, position) -> GridObject | None:
        for object in self.objects:
            if np.all(object.position == position):
                return object
        return None
    
    def load_map(self, positions):
        map = []
        for position in positions:
            map.append(ob.Wall(position))
        return map
    
    def get_state(self):
        state = np.zeros([self.width * self.height, 3])
        for object in np.concatenate((self.map_layout, self.objects, np.array([self.agent]))):
            x,y = object.position
            state[x + y*self.width] = np.array([
                OBJECT_TO_IDX[object.name],
                object.color,
                object.state,
                ])
        return state

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
        
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((255, 255, 255))

        # Draw walls
        for wall in self.map_layout:
            wall.render(canvas, self.tile_size)

        # Draw objects
        for object in self.objects:
            object.render(canvas, self.tile_size)

        # Draw Agent
        self.agent.render(canvas, self.tile_size)

        # Grid lines 
        line_color = (210, 210, 210)
        for x in range(self.height + 1):
            pygame.draw.line(
                canvas,
                line_color,
                (0, self.tile_size * x),
                (self.window_width, self.tile_size * x),
                width=3,
            )
        
        for y in range(self.width + 1):  
            pygame.draw.line(
                canvas,
                line_color,
                (self.tile_size * y, 0),
                (self.tile_size * y, self.window_height),
                width=3,
            )

        if self.render_mode == "human":
            # Copy from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
   
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
