from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class Solution:
    routes: list = None
    cost: float = 0.0
    reward: float = 0.0
    reward_after: float = 0.0

    def __post_init__(self):
        self.routes = []


class Simulation(ABC):
    @classmethod
    @abstractmethod
    def simulation(cls, solution, max_iterations, route_max_cost, var_level):
        pass
