import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from slh_framework.graph import Node, Edge, Route


def draw_route(route):
    G = nx.DiGraph()
    
    # Add nodes and edges to the graph
    for edge in route.edges:
        G.add_edge(edge.origin.id_, edge.end.id_, weight=edge.cost)
    
    pos = nx.spring_layout(G)  # positions for all nodes
    
    # Draw the graph
    node_colors = ['lightgreen' if node == route.edges[0].origin else 
                   'salmon' if node == route.edges[-1].end else 'skyblue' for node in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_size=500, node_color=node_colors, arrows=True)
    
    edge_labels = {(edge.origin.id_, edge.end.id_): edge.cost for edge in route.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.show()

def create_drone_mission():
    # Define nodes (locations) with x, y coordinates and rewards
    nodes = [Node(id_=i, x=i*1.0, y=i*1.0, reward=i*10.0) for i in range(10)]

    # Define edges (routes between locations)
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

    # Set inverse edges
    for edge in edges:
        edge.inverse_edge = Edge(origin=edge.end, end=edge.origin, cost=edge.cost)

    # Create a route and add edges
    route = Route()
    route.edges.extend(edges)
    route.cost = sum(edge.cost for edge in route.edges)
    route.reward = sum(node.reward for node in nodes)

    # Print route details
    print(route)

    # Draw the route
    draw_route(route)

if __name__ == "__main__":
    create_drone_mission()
