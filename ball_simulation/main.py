# main.py

import os
import json

from simulation import Simulation
from plotting import SimulationPlotter


def load_simulation_config(config_path="examples/config.json", mode="newtonian"):
    """
    Load config from JSON, using the given mode as a key in the JSON.
    Default path assumes 'examples/config.json' is located one directory
    above the 'ball_simulation' folder.
    """
    # Move one directory up from the folder containing this file:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_abspath = os.path.join(project_root, config_path)

    if not os.path.exists(config_abspath):
        raise FileNotFoundError(f"Configuration file not found: {config_abspath}")

    with open(config_abspath, "r") as f:
        all_config = json.load(f)

    if mode not in all_config:
        raise ValueError(f"Mode '{mode}' not found in the configuration file.")

    return all_config[mode]


def setup_simulation(config):
    """
    Create and return a Simulation object based on the loaded config.
    """
    well_radius = config["well"]["radius"]
    well_height = config["well"]["height"]
    total_time = config["simulation"]["total_time"]
    dt = config["simulation"]["dt"]
    movement_type = config["simulation"]["movement_type"]

    # Initialize the Simulation
    sim = Simulation(
        well_radius=well_radius,
        well_height=well_height,
        total_time=total_time,
        dt=dt,
        movement_type=movement_type
    )

    # Add water molecules, if any
    for molecule in config["particles"]["oxygen_molecules"]:
        center_position = molecule["center_position"]
        molecule_id = molecule.get("molecule_id", None)
        sim.create_water_molecule(center_position, molecule_id=molecule_id)

    # Add any custom particles
    for particle in config["particles"]["custom_particles"]:
        sim.add_ball(
            mass=particle["mass"],
            initial_position=particle["position"],
            initial_velocity=particle["velocity"],
            species=particle["species"]
        )

    return sim

