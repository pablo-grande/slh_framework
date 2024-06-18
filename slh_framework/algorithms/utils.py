import operator
import math
import random
import time

import numpy as np

from copy import copy, deepcopy
from slh_framework.simulations.base import Solution
from slh_framework.graph import Node, Edge, Route


def euclidean(x_1, x_2, y_1, y_2):
    return math.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2)


def timeit(func):
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Function {func.__name__!r} executed in {(t2-t1):.4f}s")
        return result

    return wrap_func


class HeuristicUtils:
    @staticmethod
    def to_node_list(list_):
        return [Node(index, *data) for index, data in enumerate(list_)]

    @staticmethod
    def generate_initial_solution(test, fleet_size, route_max_cost, nodes):
        def generate_efficiency_list(nodes, alpha):
            start = nodes[0]
            finish = nodes[-1]
            for node in nodes[1:-1]:  # excludes the start and finish depots
                sn_edge = Edge(start, node)  # creates the (start, node) edge (arc)
                nf_edge = Edge(node, finish)  # creates the (node, finish) edge (arc)
                # compute the Euclidean distance as cost
                sn_edge.cost = euclidean(start.x, node.x, start.y, node.y)
                nf_edge.cost = euclidean(finish.x, node.x, finish.y, node.y)
                # save in node a reference to the (depot, node) edge (arc)
                node.depot_to_node = sn_edge
                node.node_to_depot = nf_edge

            efficiency_list = []
            for i in range(1, len(nodes) - 2):  # excludes the start and finish depots
                i_node = nodes[i]
                for j in range(i + 1, len(nodes) - 1):
                    j_node = nodes[j]
                    ij_edge = Edge(i_node, j_node)  # creates the (i, j) edge
                    ji_edge = Edge(j_node, i_node)
                    ij_edge.inverse_edge = ji_edge  # sets the inverse edge (arc)
                    ji_edge.inverse_edge = ij_edge
                    # compute the Euclidean distance as cost
                    ij_edge.cost = euclidean(i_node.x, j_node.x, i_node.y, j_node.y)
                    ji_edge.cost = ij_edge.cost  # assume symmetric costs
                    # compute efficiency as proposed by Panadero et al.(2020)
                    ij_savings = (
                        i_node.node_to_depot.cost
                        + j_node.depot_to_node.cost
                        - ij_edge.cost
                    )
                    edge_reward = i_node.reward + j_node.reward
                    ij_edge.savings = ij_savings
                    ij_edge.efficiency = alpha * ij_savings + (1 - alpha) * edge_reward
                    ji_savings = (
                        j_node.node_to_depot.cost
                        + i_node.depot_to_node.cost
                        - ji_edge.cost
                    )
                    ji_edge.savings = ji_savings
                    ji_edge.efficiency = alpha * ji_savings + (1 - alpha) * edge_reward
                    # save both edges in the efficiency list
                    efficiency_list.append(ij_edge)
                    efficiency_list.append(ji_edge)

            # sort the list of edges from higher to lower efficiency
            efficiency_list.sort(key=operator.attrgetter("efficiency"), reverse=True)
            return efficiency_list

        best_reward = 0
        efficiency_list, initial_solution = None, None
        for alpha in np.linspace(0, 1, 11):
            new_efficiency_list = generate_efficiency_list(nodes, alpha)
            solution = HeuristicUtils.merge_routes(
                test, fleet_size, route_max_cost, nodes, new_efficiency_list
            )
            if solution.reward > best_reward:
                best_reward = solution.reward
                efficiency_list = new_efficiency_list
                initial_solution = solution

        return efficiency_list, initial_solution

    @staticmethod
    def modify_solution(current_solution, fleet_size, route_max_cost, nodes, efficiency_list):
        new_solution = deepcopy(current_solution)
        
        # select a route to modify
        route_index = random.randint(0, len(new_solution.routes) - 1)
        route = new_solution.routes[route_index]
        
        # Swap two edges within the route if there are more than one edge
        if len(route.edges) > 1:
            i, j = random.sample(range(len(route.edges)), 2)
            route.edges[i], route.edges[j] = route.edges[j], route.edges[i]
        
        # Move an edge to a different route if there is more than one route
        if len(new_solution.routes) > 1:
            from_route_index = random.randint(0, len(new_solution.routes) - 1)
            to_route_index = random.choice([i for i in range(len(new_solution.routes)) if i != from_route_index])
            from_route = new_solution.routes[from_route_index]
            to_route = new_solution.routes[to_route_index]
            
            if from_route.edges:
                edge_to_move = from_route.edges.pop(random.randint(0, len(from_route.edges) - 1))
                insertion_index = random.randint(0, len(to_route.edges))
                to_route.edges.insert(insertion_index, edge_to_move)
        
        # Calculate cost and reward for the modified solution
        for route in new_solution.routes:
            route.cost = sum(edge.cost for edge in route.edges)
            route.reward = sum(edge.end.reward for edge in route.edges if edge.end is not None)
        
        new_solution.total_cost = sum(route.cost for route in new_solution.routes)
        new_solution.total_reward = sum(route.reward for route in new_solution.routes)
        
        return new_solution
    
    @staticmethod
    def get_random_position(beta_1, beta_2, efficiency_list_size):
        beta = beta_1 + random.random() * (beta_2 - beta_1)
        index = int(math.log(random.random()) / math.log(1 - beta))
        return index % efficiency_list_size

    @staticmethod
    def merge_routes(
        test, fleet_size, route_max_cost, nodes, efficiency_list_, br=False
    ):
        def can_merge(i_node, j_node, i_route, j_route, ij_edge, route_max_cost):
            if i_route == j_route:
                return False
            if (
                i_node.is_linked_to_finish is False
                or j_node.is_linked_to_start is False
            ):
                return False
            if i_route.cost + j_route.cost - ij_edge.savings > route_max_cost:
                return False
            return True

        solution = HeuristicUtils.dummy_solution(route_max_cost, nodes)
        efficiency_list = copy(efficiency_list_)
        while len(efficiency_list) > 0:
            position = (
                HeuristicUtils.get_random_position(
                    test.first_param, test.second_param, len(efficiency_list)
                )
                if br
                else 0
            )
            ij_edge = efficiency_list.pop(position)
            i_node = ij_edge.origin
            j_node = ij_edge.end
            i_route = i_node.in_route
            j_route = j_node.in_route
            if can_merge(i_node, j_node, i_route, j_route, ij_edge, route_max_cost):
                ji_edge = ij_edge.inverse_edge
                if ji_edge in efficiency_list:
                    efficiency_list.remove(ji_edge)
                i_edge = i_route.edges[-1]
                i_route.edges.remove(i_edge)
                i_route.cost -= i_edge.cost
                i_node.is_linked_to_finish = False
                j_edge = j_route.edges[0]
                j_route.edges.remove(j_edge)
                j_route.cost -= j_edge.cost
                j_node.is_linked_to_start = False
                # add ij_edge to i_route
                i_route.edges.append(ij_edge)
                i_route.cost += ij_edge.cost
                i_route.reward += j_node.reward
                j_node.in_route = i_route
                # add j_route to new i_route
                for edge in j_route.edges:
                    i_route.edges.append(edge)
                    i_route.cost += edge.cost
                    i_route.reward += edge.end.reward
                    edge.end.in_route = i_route
                # delete j_route from emerging solution
                solution.cost -= ij_edge.savings
                solution.routes.remove(j_route)

        solution.routes.sort(key=operator.attrgetter("reward"), reverse=True)
        for route in solution.routes[fleet_size:]:
            solution.reward -= route.reward
            solution.cost -= route.cost
            solution.routes.remove(route)
        return solution

    @staticmethod
    def dummy_solution(route_max_cost, nodes):
        solution = Solution()
        # iterate all nodes except for start and finish depots
        for node in nodes[1:-1]:
            sn_edge = node.depot_to_node
            nf_edge = node.node_to_depot

            route = Route()
            route.edges.append(sn_edge)
            route.reward += node.reward
            route.cost += sn_edge.cost
            route.edges.append(nf_edge)
            route.cost += nf_edge.cost
            node.in_route = route
            node.is_linked_to_start, node.is_linked_to_finish = True, True

            if route.cost <= route_max_cost:
                solution.routes.append(route)
                solution.cost += route.cost
                solution.reward += route.reward

        return solution
