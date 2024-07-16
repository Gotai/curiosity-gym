from dataclasses import dataclass

@dataclass
class EnvironmentSettings:
    max_steps: int = 50
    width: int = 10
    heigth: int = 10
    reward_range: tuple[int,int] = (0,1)
