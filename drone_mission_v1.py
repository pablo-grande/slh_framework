"""Naive drone mission, creates edges and nodes and plots them in a grid."""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from slh_framework.graph import Node, Edge, Route

def draw_route(route, signal_quality):
    G = nx.DiGraph()
    
    for edge in route.edges:
        G.add_edge(edge.origin.id_, edge.end.id_, weight=edge.cost)
    
    pos = {node.id_: (node.x, node.y) for node in route.nodes}

    plt.imshow(signal_quality, cmap='viridis', interpolation='nearest', extent=[-0.5, 9.5, -0.5, 9.5])
    plt.colorbar(label='Signal Quality')

    node_colors = ['lightgreen' if node == route.edges[0].origin.id_ else 
                   'salmon' if node == route.edges[-1].end.id_ else 'skyblue' for node in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_size=500, node_color=node_colors, arrows=True, edge_color='white')
    
    edge_labels = {(edge.origin.id_, edge.end.id_): edge.cost for edge in route.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='blue')
    
    plt.show()

def create_drone_mission():
    signal_quality = np.random.rand(10, 10) * 10
    # for testing purposes
    signal_quality = signal_quality[:5]

    flat_indices = np.argsort(signal_quality, axis=None)[::-1]
    best_positions = np.unravel_index(flat_indices[:10], signal_quality.shape)
    
    nodes = [Node(id_=i, x=signal_quality[1][i], y=signal_quality[0][i], reward=signal_quality[best_positions[0][i], best_positions[1][i]]) for i in range(10)]

    edges = [
        Edge(origin=nodes[0], end=nodes[1], cost=5.0),
        Edge(origin=nodes[1], end=nodes[2], cost=7.0),
        Edge(origin=nodes[2], end=nodes[3], cost=8.0),
        Edge(origin=nodes[3], end=nodes[4], cost=6.0),
        Edge(origin=nodes[4], end=nodes[5], cost=3.0),
        Edge(origin=nodes[5], end=nodes[6], cost=4.0),
        Edge(origin=nodes[6], end=nodes[7], cost=2.0),
        Edge(origin=nodes[7], end=nodes[8], cost=9.0),
        Edge(origin=nodes[8], end=nodes[9], cost=1.0),
        Edge(origin=nodes[9], end=nodes[0], cost=10.0),
    ]

    for edge in edges:
        edge.inverse_edge = Edge(origin=edge.end, end=edge.origin, cost=edge.cost)

    route = Route()
    route.edges.extend(edges)
    route.cost = sum(edge.cost for edge in route.edges)
    route.reward = sum(node.reward for node in nodes)
    route.nodes = nodes

    draw_route(route, signal_quality)

if __name__ == "__main__":
    create_drone_mission()