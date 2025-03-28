# main.py
import json
import argparse
from src.simulation import Simulation
from src.plotting import SimulationPlotter
import sys # For exiting if config fails

def load_config(config_path):
    """Loads configuration from JSON file."""
    # No try-except block per user requirement
    # If file not found or invalid JSON, program will crash (as requested)
    with open(config_path, "r") as file:
        config = json.load(file)
    print(f"Configuration loaded from: {config_path}")
    return config

def setup_and_run(config):
    """Initializes simulation and plotter, then runs the animation."""
    plot_config = config.get("plotting") # Optional plotting settings

    print("Initializing simulation...")
    sim = Simulation.from_config(config) # Factory method handles setup

    print("Initializing plotter...")
    plotter = SimulationPlotter(sim, plot_config=plot_config)

    # Run the animation (includes simulation loop)
    plotter.animate_simulation() # Add save_path argument if needed

def main():
    """Parses arguments and orchestrates the simulation run."""
    parser = argparse.ArgumentParser(description="Run Molecular Dynamics Simulation")
    parser.add_argument(
        "-c", "--config",
        default="config.json",
        help="Path to the configuration JSON file (default: config.json)"
    )
    args = parser.parse_args()

    # Load configuration - program will exit on error here
    config = load_config(args.config)

    # Setup and run simulation/plotting
    setup_and_run(config)

if __name__ == "__main__":
    main()