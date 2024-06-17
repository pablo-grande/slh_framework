"""Creates a drone mission for going to the best signal quality spots (systematically)."""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from slh_framework.graph import Node, Edge, Route


# create a 10x10 grid of random signal qualities (between 0 and 1)
grid_size = 10
signal_quality = np.random.rand(grid_size, grid_size)

# Sort signal qualities from best signal to worst
best_qualities = np.argsort(signal_quality, axis=None)[::-1]
# Rebuild signal coordinates ordered by value
sorted_coords = [np.unravel_index(index, signal_quality.shape) for index in best_qualities]
sorted_values = signal_quality.flatten()[best_qualities]

# create node list
nodes = []
for index, coord in enumerate(sorted_coords[:10]):
    x, y = coord
    node_values = {
        "id_": index,
        "x": x,
        "y": y,
        "reward": sorted_values[index]
    }
    nodes.append(Node(**node_values))
    print(coord, sorted_values[index])

# and edge list
edges = []
for index, node in enumerate(nodes):
    end_node = nodes[index + 1 if index + 1 < len(nodes) else 0]
    edge_values = {
        "origin": node,
        "end": end_node,
        "cost": 1.0,  # fixed value
    }
    edges.append(Edge(**edge_values))

for edge in edges:
    edge.inverse_edge = Edge(origin=edge.end, end=edge.origin, cost=edge.cost)

route = Route()
route.edges.extend(edges)
route.cost = sum(edge.cost for edge in route.edges)
route.reward = sum(node.reward for node in nodes)
route.nodes = nodes

# create subplot to match NetworkX with signal quality grid
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(signal_quality, cmap="viridis", origin='upper', vmin=0, vmax=1)
plt.colorbar(im, ax=ax, label='Signal Quality')

# add text values to each grid tile
for i in range(grid_size):
    for j in range(grid_size):
        ax.text(j, i, f"{signal_quality[i, j]:.2f}", ha='center', va='center', color='black')

G = nx.DiGraph()
for edge in route.edges:
    G.add_edge(edge.origin.id_, edge.end.id_, weight=edge.cost)

pos = {node.id_: (node.y, node.x) for node in route.nodes}  # Note: (y, x) to match imshow
node_colors = ['lightgreen' if node == route.edges[0].origin.id_ else 'skyblue' for node in G.nodes()]
nx.draw(G, pos, node_size=500, node_color=node_colors, arrows=True, edge_color='white', ax=ax)

ax.grid(which='minor', color='black', linestyle='-', linewidth=2)

plt.title('Signal Quality Grid')
plt.show()