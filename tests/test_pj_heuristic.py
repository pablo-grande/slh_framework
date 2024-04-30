import logging
import unittest
from random import seed as random_seed
from numpy import random as np_random

from slh_framework.datasets import tests, TestInstance
from slh_framework.algorithms import pj_heuristic
from slh_framework.simulations import MonteCarlo as Simulation


logger = logging.getLogger("TestLogger")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)

file_handler = logging.FileHandler("test_results.log")
file_handler.setLevel(logging.DEBUG)
file_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_format)
logger.addHandler(file_handler)


class TestPJHeuristic(unittest.TestCase):
    test = {
        "instance_name": "p1.2.r",
        "expected_results": {
            8634452: {
                "OBD_reward_det": 275.0,
                "OBD_reward_stoch": 109.685,
                "OBS_reward_stoch": 110.79,
            },
            6117254: {
                "OBD_reward_det": 275.0,
                "OBD_reward_stoch": 117.755,
                "OBS_reward_stoch": 117.755,
            },
            6546730: {
                "OBD_reward_det": 275.0,
                "OBD_reward_stoch": 108.385,
                "OBS_reward_stoch": 112.12,
            },
            4917199: {
                "OBD_reward_det": 275.0,
                "OBD_reward_stoch": 130.16,
                "OBS_reward_stoch": 130.16,
            },
            5418507: {
                "OBD_reward_det": 275.0,
                "OBD_reward_stoch": 113.975,
                "OBS_reward_stoch": 117.48,
            },
            3848834: {
                "OBD_reward_det": 275.0,
                "OBD_reward_stoch": 104.875,
                "OBS_reward_stoch": 118.035,
            },
            7375358: {
                "OBD_reward_det": 275.0,
                "OBD_reward_stoch": 119.995,
                "OBS_reward_stoch": 119.995,
            },
            7664268: {
                "OBD_reward_det": 275.0,
                "OBD_reward_stoch": 110.875,
                "OBS_reward_stoch": 118.5,
            },
            9150967: {
                "OBD_reward_det": 275.0,
                "OBD_reward_stoch": 108.72,
                "OBS_reward_stoch": 112.4,
            },
            1025747: {
                "OBD_reward_det": 275.0,
                "OBD_reward_stoch": 120.85,
                "OBS_reward_stoch": 120.85,
            },
        },
    }
    expected_results = test["expected_results"]
    predetermined_seeds = list(expected_results.keys())

    def test_monte_carlo(self):
        test = tests[self.test["instance_name"]]
        for seed in self.predetermined_seeds:
            random_seed(seed)
            np_random.seed(seed)

            Simulation.condition_factors = {
                "weather": {"factor": 0.2},
                "traffic": {"factor": 0.3},
                # other factors may be added
            }
            OBD, OBS = pj_heuristic(test, test.instance_data, Simulation.simulation)

            expected = self.expected_results[seed]
            self.assertAlmostEqual(
                OBD.reward,
                expected["OBD_reward_det"],
                f"Failed at seed {seed}: OBD.reward in deterministic environment",
            )
            self.assertAlmostEqual(
                OBD.reward_after,
                expected["OBD_reward_stoch"],
                msg=f"Failed at seed {seed}: OBD.reward after in stochastic environment",
            )
            self.assertAlmostEqual(
                OBS.reward_after,
                expected["OBS_reward_stoch"],
                msg=f"Failed at seed {seed}: OBS.reward after in stochastic environment",
            )


class TestPJHeuristicMultipleConditions(unittest.TestCase):
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

    def test_other_conditions(self):

        random_seed(self.seed)
        np_random.seed(self.seed)

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
            self.test, self.test.instance_data, Simulation.simulation
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


if __name__ == "__main__":
    unittest.main()
