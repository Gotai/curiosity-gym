from enum import Enum

class Action(Enum):
    FORWARD = 0
    TURN_RIGHT = 1
    TURN_LEFT = 2
    INTERACT = 3

class Rotation(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3
