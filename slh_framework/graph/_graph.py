from dataclasses import dataclass


@dataclass
class Node:
    """Node object.

    Attributes:
        id_ (int): The node identifier.
        x (float): The euclidean x-coordinate of the node.
        y (float): The euclidean y-coordinate of the node.
        reward (float): The reward associated with the node.
        in_route (Route): The route to which the node belongs.
        depot_to_node_edge (Edge): The arc from the start depot to this node.
        node_to_depot_edge (Edge): The arc from this node to the finish depot.
        is_linked_to_start (bool): Indicates whether the node is linked to the start depot.
        is_linked_to_finish (bool): Indicates whether the node is linked to the finish depot.
    """

    id_: int
    x: float
    y: float
    reward: float
    is_linked_to_start: bool = False
    is_linked_to_finish: bool = False
    in_route = None
    depot_to_node = None
    node_to_depot = None


@dataclass
class Edge:
    """
    Initialize an Edge object.

    Attributes:
        origin (Node): The origin node of the edge (arc).
        end (Node): The end node of the edge (arc).
        cost (float): The edge cost (e.g., travel time, monetary cost, etc.).
        savings (float): The edge savings (Clarke & Wright).
        inverse_edge (Edge): The inverse edge (arc).
        efficiency (float): The edge efficiency (enriched savings).
        type_ (int): The type of the edge. 0 = deterministic (default), 1 = stoch, 2 = dynamic.
    """

    origin: Node
    end: Node
    cost: float = 0.0
    savings: float = 0.0
    efficiency: float = 0.0
    inverse_edge = None
    type_ = None


@dataclass
class Route:
    """
    Attributes:
        cost (float): cost of this route
        edges (list): sorted edges in this route
        reward (float): total reward collected in this route
    """
    edges: list = None
    cost: float = 0.0
    reward: float = 0.0

    def __post_init__(self):
        self.edges = []

    def reverse(self):
        # e.g. 0 -> 2 -> 6 -> 0 becomes 0 -> 6 -> 2 -> 0
        size = len(self.edges)
        for i in range(size):
            edge = self.edges[i]
            inverse_edge = edge.inverse_edge
            self.edges.remove(edge)
            self.edges.insert(0, inverse_edge)

    def __str__(self):
        route_path = "0" + ''.join(f" -> {edge.end.id_}" for edge in self.edges)
        total_reward = f"Route det. reward: {self.reward}; det. cost: {self.cost}"
        return f"{route_path}\n{total_reward}"