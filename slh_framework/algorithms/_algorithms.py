import math
import random

from collections import deque
from time import time

from slh_framework.algorithms.utils import HeuristicUtils


def simulated_annealing_heuristic(test_instance, test_data, simulation):
    fleet_size = test_data["fleet_size"]
    route_max_cost = test_data["route_max_cost"]
    nodes = HeuristicUtils.to_node_list(test_data["node_list"])
    
    efficiency_list, initial_solution = HeuristicUtils.generate_initial_solution(
        test_data, fleet_size, route_max_cost, nodes
    )
    simulation(
        initial_solution,
        test_instance.short_sim,
        route_max_cost,
        test_instance.var_level,
    )
    
    current_solution = initial_solution
    best_solution = initial_solution
    best_reward = initial_solution.reward
    
    temperature = test_instance.initial_temp
    alpha = test_instance.cooling_rate
    min_temp = test_instance.min_temp
    
    start_time = time()
    elapsed = 0
    
    while elapsed < test_instance.max_time and temperature > min_temp:
        new_solution = HeuristicUtils.modify_solution(
            current_solution, fleet_size, route_max_cost, nodes, efficiency_list
        )
        
        simulation(
            new_solution,
            test_instance.short_sim,
            route_max_cost,
            test_instance.var_level,
        )
        
        delta_reward = new_solution.reward - current_solution.reward
        
        if delta_reward > 0 or random.uniform(0, 1) < math.exp(delta_reward / temperature):
            current_solution = new_solution
            if new_solution.reward > best_reward:
                best_solution = new_solution
                best_reward = new_solution.reward
        
        temperature *= alpha
        elapsed = time() - start_time
    
    simulation(best_solution, test_instance.long_sim, route_max_cost, test_instance.var_level)

    return best_solution


def pj_heuristic(test_instance, test_data, simulation):
    fleet_size = test_data["fleet_size"]
    route_max_cost = test_data["route_max_cost"]
    nodes = HeuristicUtils.to_node_list(test_data["node_list"])
    # generate an efficiency list and initial solution using the best alpha value
    efficiency_list, initial_solution = HeuristicUtils.generate_initial_solution(
        test_data, fleet_size, route_max_cost, nodes
    )
    simulation(
        initial_solution,
        test_instance.short_sim,
        route_max_cost,
        test_instance.var_level,
    )
    # set initial solution as OBDF and OBS solutions
    OBD = initial_solution
    OBS = initial_solution
    # define a set of elite stochastic solutions to consider
    elite_solutions = deque(maxlen=10)
    elite_solutions.append(OBS)

    # search for better deterministic and stochastic solutions
    elapsed = 0
    start_time = time()
    while elapsed < test_instance.max_time:
        # merge process of the PJs heuristics to generate new deterministic solution
        new_solution = HeuristicUtils.merge_routes(
            test_instance, fleet_size, route_max_cost, nodes, efficiency_list, br=True
        )
        # save new best solution
        if new_solution.reward > OBD.reward:
            OBD = new_solution
        if new_solution.reward > OBS.reward:
            # simulate new deterministic solution in stochastic environment
            simulation(
                new_solution,
                test_instance.short_sim,
                route_max_cost,
                test_instance.var_level,
            )
            # update OBS solution if appropiate
            if new_solution.reward_after > OBS.reward_after:
                OBS = new_solution
                elite_solutions.append(new_solution)
        elapsed = time() - start_time

    # simulate elite solutions in stochastic environment
    simulation(OBD, test_instance.long_sim, route_max_cost, test_instance.var_level)
    OBS = OBD
    for elite_solution in elite_solutions:
        simulation(
            elite_solution,
            test_instance.long_sim,
            route_max_cost,
            test_instance.var_level,
        )
        if elite_solution.reward_after > OBS.reward_after:
            OBS = elite_solution
    return OBD, OBS
