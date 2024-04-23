from .base import Solution
from .experimental import MonteCarlo

pool = {"MonteCarlo": MonteCarlo.simulation}

__all__ = [Solution, pool]
