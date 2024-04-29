import unittest
from random import seed as random_seed
from numpy import random as np_random

from slh_framework.datasets import tests
from slh_framework.algorithms import pj_heuristic
from slh_framework.simulations import MonteCarlo as Simulation

expected_results = {
    8634452: {
        "OBD_reward_det": 275.0,
        "OBD_reward_stoch": 109.685,
        "OBS_reward_stoch": 110.79,
    },
    # Add similar entries for other seeds
}


class TestPJHeuristic(unittest.TestCase):
    predetermined_seeds = [
        8634452
    ]  # , 6117254, 6546730, 4917199, 5418507, 3848834, 7375358, 7664268, 9150967, 1025747]
    selected_test = "p1.2.r"

    def test_monte_carlo(self):
        test = tests[self.selected_test]
        for seed in self.predetermined_seeds:
            random_seed(seed)
            np_random.seed(seed)

            print(f"Testing with seed: {seed}")
            Simulation.condition_factors = {
                "weather": {"factor": 0.2},
                "traffic": {"factor": 0.3},
                # other factors may be added
            }
            OBD, OBS = pj_heuristic(test, test.instance_data, Simulation.simulation)

            expected = expected_results[seed]
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


if __name__ == "__main__":
    unittest.main()
