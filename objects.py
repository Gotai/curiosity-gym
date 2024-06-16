import numpy as np
import pygame
from abc import ABC, abstractmethod
from constants import ROTATION_TO_STATE, IX_TO_COLOR

class GridObject(ABC):
    
    def __init__(self, position, name, color=0, state=0):
        self.start_position = np.array(position)
        self.position = np.array(position)
        self.name = name
        self.start_color = color
        self.color = color
        self.start_state = state
        self.state = state

    def reset(self):
        self.position = self.start_position
        self.state = self.start_state
        self.color = self.start_color

    def step(self, action):
        pass

    def walkable(self):
        return False
    
    def interact(self, agent):
        pass

    @abstractmethod
    def render(self, canvas, pixelsquare):
        pass



class Agent(GridObject):

    def __init__(self, position, rotation):
        super().__init__(position, "agent")
        self.start_rotation = np.array(rotation)
        self.rotation = np.array(rotation)

    def rotate(self, dir):
        self.rotation = np.flip(self.rotation) * dir
        self.state = ROTATION_TO_STATE[tuple(self.rotation)]

    def reset(self):
        super().reset()
        self.rotation = self.start_rotation

    def render(self, canvas, pixelsquare):
        # Constant Parameters
        c = np.array([0.5, 0.5])
        d = 0.35
        r = self.rotation

        # Calculate points for triangle rotation
        p1 = (self.position + c - np.array([-d, d]) * r) * pixelsquare
        p2 = (self.position + c - [d, -d] * np.flip(r) + [-d, d] * r ) * pixelsquare
        p3 = (self.position + c + [d, -d] * np.flip(r) + [-d, d] * r ) * pixelsquare

        # Draw agent
        pygame.draw.polygon(
            canvas,
            IX_TO_COLOR[self.color],
            (p1, p2, p3),
            0 # filled triangle
        )


class Wall(GridObject):

    def __init__(self, position):
        super().__init__(position, "wall")

    def render(self, canvas, pixelsquare):
        pygame.draw.rect(
            canvas,
            (210, 210, 210),
            pygame.Rect(
                pixelsquare * self.position,
                (pixelsquare, pixelsquare),
            ),
        )


class Target(GridObject):
    
    def __init__(self, position):
        super().__init__(position, "target")

    def walkable(self):
        return True

    def render(self, canvas, pixelsquare):
        pygame.draw.rect(
            canvas,
            (41, 191, 18),
            pygame.Rect(
                pixelsquare * self.position,
                (pixelsquare, pixelsquare),
            ),
        )


class Door(GridObject):
    def __init__(self, position, state=1, color=0):
        super().__init__(position, "door", state=state, color=color)

    def walkable(self):
        return True if self.state == 0 else False
    
    def interact(self, agent):
        if self.state == 2 and agent.color != self.color:
            return
        
        elif self.state == 2 and agent.color == self.color:
            self.state = 0
            agent.color = agent.start_color
        
        else:
            self.state = (self.state + 1) % 2

    def render(self, canvas, pixelsquare):        
        if self.state > 0:
            pygame.draw.rect(
                canvas,
                IX_TO_COLOR[self.color],
                pygame.Rect(
                    pixelsquare * self.position,
                    (pixelsquare, pixelsquare),
                ), 5
            )

            pygame.draw.circle(
                canvas,
                IX_TO_COLOR[self.color],
                (self.position + np.array([0.75,0.5])) * pixelsquare,
                0.1 * pixelsquare,
            )
        
        else: 
            pygame.draw.rect(
                canvas,
                IX_TO_COLOR[self.color],
                pygame.Rect(
                    pixelsquare * self.position,
                    (pixelsquare*0.2, pixelsquare),
                ), 5
            )

class Key(GridObject):
    def __init__(self, position, state=0, color=0):
        super().__init__(position, "key", state=state, color=color)

    def walkable(self):
        return True if self.state == 1 else False
    
    def interact(self, agent):
        if self.state == 0:
            agent.color = self.color
            self.state = 1

    def render(self, canvas, pixelsquare):
        if self.state==0:
            pygame.draw.circle(
                canvas,
                IX_TO_COLOR[self.color],
                (self.position + np.array([0.5,0.4])) * pixelsquare,
                0.15 * pixelsquare,
                5
            )

            pygame.draw.line(
                canvas,
                IX_TO_COLOR[self.color],
                (self.position + np.array([0.5,0.55])) * pixelsquare,
                (self.position + np.array([0.5,0.8])) * pixelsquare,
                5
            )

            pygame.draw.line(
                canvas,
                IX_TO_COLOR[self.color],
                (self.position + np.array([0.5,0.7])) * pixelsquare,
                (self.position + np.array([0.4,0.7])) * pixelsquare,
                5
            )