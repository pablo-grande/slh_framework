# slh_framework

A Python package dedicated to developing and testing Sim-Learn-Heuristic (SLH) algorithms. This framework provides a robust environment for creating and experimenting with heuristic solutions, leveraging a rich dataset and a variety of simulation techniques.

Based on [Sim-learn-heuristic-TOP](https://github.com/ICSO-IN3/Sim-learn-heuristic-TOP)

## Features

- **Modular Design:** Includes separate modules for algorithms, datasets, graphs, and simulations to facilitate development and testing.
- **Rich Dataset Collection:** Over 300 datasets representing routes between nodes to test and optimize heuristic solutions.
- **Advanced Simulation Capabilities:** Supports multiple types of simulations including base, experimental, and Monte Carlo simulations to assess heuristic effectiveness under different scenarios.
- **Comprehensive Heuristic Utilities:** Tools for common tasks such as route merging and generating dummy solutions.


### Key Components

- **Graph:** Contains data classes `Node`, `Edge`, and `Route` which are essential for constructing solutions.
- **Datasets:** Features over 300 datasets along with classes `TestInstance` for setting up test parameters and `TxtFileParser` for reading dataset files.
- **Simulations:** Includes `Simulation` base class, `ExperimentalSimulation`, and `MonteCarlo` simulation for diverse experimental setups.
- **Algorithms:** Provides `HeuristicUtils` for general heuristic functions and `pj_heuristic` which combines test instances and simulations for solution generation.

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/slh_framework.git
cd slh_framework
pip install -e .
```

## Usage
Example of running a heuristic algorithm with Monte Carlo simulation:
```python
from random import seed
from numpy import random as np_random

from slh_framework.datasets import tests
from slh_framework.algorithms import pj_heuristic
from slh_framework.simulations import MonteCarlo

test = tests["p1.2.r"]
seed(test.seed)
np_random.seed(test.seed)

OBD, OBS = pj_heuristic(test, test.instance_data, MonteCarlo.simulation)
print(f"Instance: {test.instance_name}, Variation level: {test.var_level}")
print(f"Reward for OBD solution in a Deterministic environment = {OBD.reward}")
print(f"Reward for OBD solution in a Stochastic environment = {OBD.reward_after}")
print(f"Reward for OBS solution in a Stochastic environment = {OBS.reward_after}")

print('Routes for OBD solution')
for route in OBD.routes:
    print(route)
print('Routes for OBS solution')
for route in OBS.routes:
    print(route)
```
