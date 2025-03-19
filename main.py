import json
from src.simulation import Simulation
from src.plotting import SimulationPlotter

def main(config_path="config.json"):
    # Load simulation parameters from config_one_water.json
    with open(config_path, "r") as file:
        config = json.load(file)

    # Create the Simulation object from the configuration.
    # This will create the water molecules (two, as defined in config)
    sim = Simulation.from_config(config)

    # Create the plotter and animate the simulation.
    plotter = SimulationPlotter(sim)
    plotter.animate_simulation()

if __name__ == "__main__":
    main()
