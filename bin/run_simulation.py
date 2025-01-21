#!/usr/bin/env python3

import os
import json
from ball_simulation.simulation import Simulation
from ball_simulation.input_output import load_config


def save_output(output_data, output_path):
    """
    Saves the simulation output to a JSON file.

    Args:
        output_data (dict): The simulation data to save.
        output_path (str): The path to the output file.

    Raises:
        FileNotFoundError: If the directory for the output file does not exist.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save the output to the specified file
    with open(output_path, 'w') as file:
        json.dump(output_data, file, indent=4)
    print(f"Simulation output saved to: {output_path}")


def main(config_path=None, output_path=None):
    """
    Main function to set up and run the simulation.

    Args:
        config_path (str, optional): Path to the configuration file.
                                     Defaults to 'examples/config.json' relative to the project root.
        output_path (str, optional): Path to save the simulation output.
                                     Defaults to 'examples/simulation_output.json' relative to the project root.

    Steps:
        1. Resolves the absolute paths of the configuration and output files.
        2. Loads configuration parameters from the JSON file.
        3. Sets up the simulation using the `Simulation` class.
        4. Runs the simulation and saves the output to a file.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    # Resolve the absolute paths of the configuration and output files
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if config_path is None:
        config_path = os.path.join(project_root, "examples", "config.json")
    if output_path is None:
        output_path = os.path.join(project_root, "examples", "simulation_output.json")

    # Debugging: Print the resolved paths
    print(f"Using configuration file: {config_path}")
    print(f"Saving simulation output to: {output_path}")

    # Load the configuration file
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    config = load_config(config_path)

    # Extract simulation parameters
    well_radius = config["well"]["radius"]
    well_height = config["well"]["height"]
    total_time = config["simulation"]["total_time"]
    dt = config["simulation"]["dt"]
    movement_type = config["simulation"]["movement_type"]

    # Set up and run the simulation
    sim = Simulation(well_radius=well_radius, well_height=well_height, total_time=total_time, dt=dt)
    sim.set_movement_type(movement_type)

    # Add balls to the simulation based on configurations
    for ball_config in config["balls"]:
        sim.add_ball(
            mass=ball_config["mass"],
            initial_position=ball_config["initial_position"],
            initial_velocity=ball_config["initial_velocity"],
        )

    # Run the simulation
    print("Running simulation...")
    sim.run()

    # Collect results, including path segments
    simulation_results = {
        "balls": [
            {
                "position": ball.position.tolist(),
                "velocity": ball.velocity.tolist(),
                "mass": ball.mass,
                "path_segments": ball.get_path_segments()  # Include path segments
            }
            for ball in sim.balls
        ],
        "well": {
            "radius": sim.well.radius,
            "height": sim.well.height
        },
        "simulation_parameters": {
            "total_time": sim.total_time,
            "dt": sim.dt,
            "movement_type": sim.movement_type,
        }
    }

    # Save the simulation results to the output file
    save_output(simulation_results, output_path)

    print("Simulation complete. Results have been saved.")


if __name__ == "__main__":
    # Entry point for the script
    main()
