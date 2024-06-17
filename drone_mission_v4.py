"""Creates a drone mission with SLH framework heuristic components."""
import logging

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from random import seed as random_seed

from slh_framework.datasets import TestInstance
from slh_framework.algorithms import pj_heuristic
from slh_framework.simulations import MonteCarlo as Simulation


logger = logging.getLogger("Drone mission")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)


def create_map_and_coords(grid_size):
    """Create a 'grid_size' map (grid) of random signal qualities between 0 and 1."""
    signal_quality = np.random.rand(grid_size, grid_size)
    coords = [(i, j) for i in range(grid_size) for j in range(grid_size)]
    np.random.shuffle(coords)
    selected_coords = coords[:number_of_nodes]
    return signal_quality, selected_coords


def plot(signal_quality, route, filename=None):
    """Make transformations on route to print results onto signal_quality grid."""
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(signal_quality, cmap="viridis", origin='upper', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Signal Quality')

    # add text values to each grid tile
    grid_size = signal_quality.shape[0]
    for i in range(grid_size):
        for j in range(grid_size):
            ax.text(j, i, f"{signal_quality[i, j]:.2f}", ha='center', va='center', color='black')

    positions = {}
    G = nx.DiGraph()
    for edge in route.edges:
        G.add_edge(edge.origin.id_, edge.end.id_, weight=edge.cost)
        positions[edge.origin.id_] = (edge.origin.y, edge.origin.x)
        positions[edge.end.id_] = (edge.end.y, edge.end.x)

    # now we can paint the position of each node
    node_colors = ['lightgreen' if node == route.edges[0].origin.id_ else 'skyblue' for node in G.nodes()]
    nx.draw(G, positions, node_size=500, node_color=node_colors, arrows=True, edge_color='white', ax=ax)

    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)

    plt.title('Signal Quality Grid')
    if filename is not None:
        plt.savefig(filename)
    else: 
        plt.show()


map_size = 10
number_of_nodes = 10
seed = 1025747
test = TestInstance("drone_sweep")
test.instance_data["fleet_size"] = 2
test.instance_data["route_max_cost"] = 42.5

random_seed(seed)
np.random.seed(seed)

Simulation.condition_factors = {
    "weather": {"factor": 0.2},
    "unexplored_area": {"factor": 0.4},
    # in LTE network, a UE measures two parameters on reference signal: 
    # * RSRP (Reference Signal Received Power)
    # * RSRQ (Reference Signal Received Quality)
    "rsrq": {"factor": 0.5},
    "rsrp": {"factor": 0.1},
}


signal_quality, selected_coords = create_map_and_coords(map_size)
node_list = []
for index, coord in enumerate(selected_coords):
    x, y = coord
    node_list.append(
        (x, y, signal_quality[x, y])
    )
test.instance_data["node_list"] = node_list
test.instance_data["number_of_nodes"] = number_of_nodes

logger.info("Starting mission")

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

selected_route = OBS.routes[0]
plot(signal_quality, selected_route, filename="OBS route")
