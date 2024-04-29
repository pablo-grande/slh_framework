import os

from random import randint


SEED_SIZE = 7


class TxtFileParser:
    meta = {
        "n": ("number_of_nodes", int),
        "m": ("fleet_size", int),
        "tmax": ("route_max_cost", float),
    }

    def __init__(self, filepath):
        self.filepath = filepath
        self.data = {}

    def parse_file(self):
        """Read the file and parse each line using parse_line method."""
        nodes = []
        with open(self.filepath, "r") as file:
            for line in file:
                key, value = self.parse_line(line)
                if key and value:  # Ensure that both key and value are not None
                    # check for metadata values
                    if key in self.meta:
                        key_name, key_type = self.meta[key]
                        self.data[key_name] = key_type(value)
                    else:
                        # it's a node
                        nodes.append(tuple([float(x) for x in line.split(";")]))
        self.data["node_list"] = nodes

    def parse_line(self, line, separator=";"):
        """Parse individual lines; can be overridden for specialized line types."""
        line = line.strip()
        if line:
            parts = line.split(separator, 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                return key, value
        return None, None  # Return None if the line doesn't match the expected format

    def __repr__(self):
        return f"<{self.__class__.__name__} ({self.filepath})>: {self.data}"


SelectedParser = os.getenv("parser", TxtFileParser)


class TestInstance:
    """
    The Test class represents a test instance with various parameters.

    Attributes:
        instance_name (str): The name of the test instance.
        max_time (int): The maximum computation time in seconds.
        first_param (float): The lower bound for beta in Geom(beta).
        second_param (float): The upper bound for beta in Geom(beta).
        seed (int): The seed for the RNG for reproducibility purposes.
        short_sim (int): The number of runs in a short simulation.
        long_sim (int): The number of runs in a long simulation.
        var_level (float): The variance level.
        filename (str): The filepath of the test instance.
    """

    def __init__(
        self,
        instance_name="",
        max_time=60,
        first_param=0.1,
        second_param=0.3,
        seed=None,
        short_sim=100,
        long_sim=1000,
        var_level=1.0,
        filename=None,
    ):
        self.instance_name = instance_name
        self.max_time = int(max_time)
        self.first_param = float(first_param)
        self.second_param = float(second_param)
        if seed is None:
            lower_bound = 10 ** (SEED_SIZE - 1)
            upper_bound = (10**SEED_SIZE) - 1
            seed = randint(lower_bound, upper_bound)
        self.seed = int(seed)
        self.short_sim = int(short_sim)
        self.long_sim = int(long_sim)
        self.var_level = float(var_level)
        self.instance_data = {
            "number_of_nodes": 0,
            "fleet_size": 0,
            "route_max_cost": 0.0,
            "node_list": [],
        }
        if filename is not None:
            file_parser = SelectedParser(filename)
            file_parser.parse_file()
            self.instance_data = file_parser.data
            self.instance_name = filename.split(os.path.sep)[-1].split(".txt")[0]

    def __repr__(self):
        return f"<{self.__class__.__name__}>: {self.__dict__}"


_dirname = os.path.dirname(__file__)
tests = {}
for file in os.listdir(_dirname):
    if os.path.isfile(os.path.join(_dirname, file)) and file.endswith(".txt"):
        test = TestInstance(filename=f"{_dirname}{os.path.sep}{file}")
        tests[test.instance_name] = test
