import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame

class GridEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, agentview=None, size=5, windowsize=512, render_mode=None, max_steps=30):
        self.size = size
        self.agent_view = agentview
        #self.action_space = self.agent_view.get_action_space()
        self.window_size = windowsize
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.max_steps = max_steps

    def _get_obs(self):
        #return self.agent_view.get_obs()
        return {"agent_pos": self._agent_location, "agent_rot": self._agent_rotation, "target": self._target_location}

    # probably by overloaded by concrete env
    def reset(self):
        self.steps = 0
        self._agent_location = np.array([0,0]) # x, y coordinates in grid
        self._agent_rotation = np.array([1,0]) # rotation (horizontal, vertical)
        self._target_location = np.array([5,5]) # target location x, y

        return self._get_obs()
    
    def step(self, action):
        
        self.steps += 1

        if action == 0:
            self._agent_location = np.clip(
                self._agent_location + self._agent_rotation * np.array([1,-1]), 0, self.size-1)

        elif action == 1:
            self._agent_rotation = np.array([self._agent_rotation[1], self._agent_rotation[0]*-1])

        elif action == 2:
            self._agent_rotation = np.array([self._agent_rotation[1]*-1, self._agent_rotation[0]])

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_reward(), self._get_terminated(), False, []
    
    def _get_reward(self):
        return 1 if np.array_equal(self._agent_location, self._target_location) else 0
    
    def _get_terminated(self):
        return True if np.array_equal(self._agent_location, self._target_location) or self.steps >= self.max_steps else False

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()


    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Draw target
        pygame.draw.rect(
            canvas,
            (41, 191, 18),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )

        # Calculate points for triangle rotation
        c = np.array([0.5, 0.5])
        d = 0.35
        r = self._agent_rotation
        p1 = (self._agent_location + c - np.array([-d, d]) * r) * pix_square_size
        p2 = (self._agent_location + c - [d, -d] * np.flip(r) + [-d, d] * r ) * pix_square_size
        p3 = (self._agent_location + c + [d, -d] * np.flip(r) + [-d, d] * r ) * pix_square_size

        # Draw agent
        pygame.draw.polygon(
            canvas,
            (255,153,20),
            (p1, p2, p3),
            0 # filled triangle
        )

        # Grid lines 
        line_color = (210, 210, 210)
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                line_color,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                line_color,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
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