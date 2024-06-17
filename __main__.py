import logging

from random import seed as random_seed
from numpy import random as np_random

from slh_framework.datasets import tests, TestInstance
from slh_framework.algorithms import pj_heuristic
from slh_framework.simulations import MonteCarlo as Simulation


logger = logging.getLogger("Main")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)


def draw_routes(routes, prepend_name):
    for index, route in enumerate(routes, start=1):
        route.draw(f"{prepend_name}_{1}")


seed = 1025747
test = TestInstance("drone_sweep")
test.instance_data["fleet_size"] = 2
test.instance_data["route_max_cost"] = 42.5
node_list = [
    (10.5, 14.4, 0.0),
    (18.0, 15.9, 10.0),
    (18.3, 13.3, 10.0),
    (16.5, 9.3, 10.0),
    (15.4, 11.0, 10.0),
    (14.9, 13.2, 5.0),
    (16.3, 13.3, 5.0),
    (16.4, 17.8, 5.0),
    (15.0, 17.9, 5.0),
    (16.1, 19.6, 10.0),
    (15.7, 20.6, 10.0),
    (13.2, 20.1, 10.0),
    (14.3, 15.3, 5.0),
    (14.0, 5.1, 10.0),
]
test.instance_data["node_list"] = node_list
test.instance_data["number_of_nodes"] = len(node_list)

random_seed(seed)
np_random.seed(seed)

Simulation.condition_factors = {
    "weather": {"factor": 0.2},
    "unexplored_area": {"factor": 0.4},
    # in LTE network, a UE measures two parameters on reference signal: 
    # * RSRP (Reference Signal Received Power)
    # * RSRQ (Reference Signal Received Quality)
    "rsrq": {"factor": 0.5},
    "rsrp": {"factor": 0.1},
}
OBD, OBS = pj_heuristic(
    test, test.instance_data, Simulation.simulation
)
logger.info(
    f"""
    OBD solution:
    cost: {OBD.cost}
    reward: {OBD.reward}
    reward after: {OBD.reward_after}
    route: {"".join(str(route) for route in OBD.routes)}
    """
)


logger.info(
    f"""
    OBS solution:
    cost: {OBS.cost}
    reward: {OBS.reward}
    reward after: {OBS.reward_after}
    route: {"".join(str(route) for route in OBS.routes)}
    """
)

draw_routes(OBS.routes, "OBS")