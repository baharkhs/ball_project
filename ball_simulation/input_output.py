# ball_simulation/input_output.py

import json


def load_config(filename):
    """
    Loads the configuration from a JSON file.

    Args:
        filename (str): The path to the JSON configuration file.

    Returns:
        dict: A dictionary with the configuration data.
    """
    with open(filename, 'r') as file:
        config = json.load(file)
    return config