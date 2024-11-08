# ball_simulation/input_output.py

import json

def load_parameters(file_path):
    """Load simulation parameters from a JSON file."""
    with open(file_path, 'r') as file:
        parameters = json.load(file)
    return parameters

def save_results(results, file_path):
    """Save simulation results to a JSON file."""
    with open(file_path, 'w') as file:
        json.dump(results, file, indent=4)
