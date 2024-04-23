import math
import numpy as np

from enum import auto, Enum
from slh_framework.simulations.base import Simulation


class EdgeType(Enum):
    DETERMINISTIC = auto()
    STOCHASTIC = auto()
    DYNAMIC = auto()

    @classmethod
    def default(cls):
        return cls.DETERMINISTIC


class ExperimentalSimulation(Simulation):
    @classmethod
    def set_edges_type(cls, solution):
        # check for divisibilty on node id
        divisible_by = {2: EdgeType.STOCHASTIC, 3: EdgeType.DYNAMIC}
        for route in solution.routes:
            for edge in route.edges:
                for quotient in divisible_by:
                    if edge.end.id_ % quotient == 0:
                        edge.type_ = divisible_by[quotient]
                        break
                else:
                    edge.type_ = EdgeType.default()

    @classmethod
    def get_stochastic_value(cls, mean=None, var_level=None, scale=None, location=None):
        if scale is None and location is None:
            var = var_level * mean
            mu = math.log(mean**2 / math.sqrt(var + mean**2))
            sigma = math.sqrt(math.log(1 + var / mean**2))
            stochastic_cost = np.random.lognormal(mean=mu, sigma=sigma)
        elif mean is None and var_level is None:
            stochastic_cost = np.random.lognormal(mean=scale, sigma=location)
        else:
            raise ValueError("Error using lognormal distribution")
        return stochastic_cost

    @classmethod
    def get_dynamic_value(cls, edge, condition_factors):
        """
        Calculate the dynamic cost of an edge based on various condition factors.

        Args:
            edge (Edge): The edge for which to calculate the cost.
            condition_factors (dict): A dictionary of condition factors where each key is a 
                                    condition name and the value is another dictionary 
                                    containing 'factor' and 'value' which are the multiplier 
                                    to the edge cost and the environmental condition value, respectively.

        Returns:
            float: The dynamically calculated cost of the edge.
        """
        b0 = 0  # independent term in a regression model, 0 so that cost in good conditions is the standard one
        b_e = 1  # coefficient for edge standard cost
        dynamic_cost = b0 + b_e * edge.cost

        # Iterate through each condition factor and apply it to the cost calculation
        for condition, factor_value in condition_factors.items():
            factor = factor_value["factor"]
            value = factor_value["value"]
            dynamic_cost += factor * edge.cost * value

        return dynamic_cost


class MonteCarlo(ExperimentalSimulation):
    @classmethod
    def simulation(cls, solution, max_iterations, route_max_cost, var_level):
        cls.set_edges_type(solution)
        accumlated_reward = 0
        for _ in range(max_iterations):
            condition_factors = {
                "condition_1": {
                    "factor": 0.2,
                    "value": np.random.random()
                }
            }
            reward_in_solution = 0
            for route in solution.routes:
                route_reward, route_cost = 0, 0
                for edge in route.edges:
                    node = edge.end
                    route_reward += node.reward
                    if edge.type_ == EdgeType.DETERMINISTIC:
                        edge_cost = edge.cost
                    elif edge.type_ == EdgeType.STOCHASTIC:
                        edge_cost = cls.get_stochastic_value(
                            mean=edge.cost, var_level=var_level
                        )
                    elif edge.type_ == EdgeType.DYNAMIC:
                        condition_factors["condition_2"] = {
                            "factor": 0.3,
                            "value": np.random.random()
                        }
                        edge_cost = super(MonteCarlo, cls).get_dynamic_value(edge, condition_factors)
                    route_cost += edge_cost

                if route_cost > route_max_cost:
                    route_reward = 0
                reward_in_solution += route_reward

            accumlated_reward += reward_in_solution
        solution.reward_after = accumlated_reward / max_iterations
