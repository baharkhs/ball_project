import sys
import os

from ball_simulation.simulation import Simulation
from ball_simulation.plotting import SimulationPlotter
from setup import load_simulation_config, setup_simulation

def main():
    """
    Main function to run the water molecule simulation based on the configuration file.
    """
    try:
        # 1) Load configuration from "examples/config.json" with default mode "newtonian"
        config_path = os.path.join("examples", "config.json")
        config = load_simulation_config(config_path=config_path, mode="newtonian")

        # 2) Set up the simulation based on the config
        sim = setup_simulation(config)

        # 3) Run the simulation for a few steps to demonstrate
        print("Running simulation for 5 steps...")
        for _ in range(5):
            sim.update()

        # 4) Print some basic simulation state
        print(f"Number of balls: {len(sim.balls)}")
        print(f"Temperature history so far: {sim.temperature_history}")

        # If you have molecule center-of-mass info, you can print it here (if your code populates it)
        if hasattr(sim, "molecule_com"):
            for mol_id in sim.molecules:
                com_data = sim.molecule_com[mol_id]
                print(f"Molecule {mol_id} COM position: {com_data['position']}, velocity: {com_data['velocity']}")
        else:
            print("Molecule COM data not found or not implemented.")

        # 5) Visualize the simulation using SimulationPlotter
        plotter = SimulationPlotter(sim)
        plotter.animate_simulation()

    except FileNotFoundError as e:
        print(f"Error: Configuration file not found - {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Invalid configuration - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
