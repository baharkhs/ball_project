import json
from pathlib import Path
from src.simulation import Simulation
from src.plotting import SimulationPlotter

def main():
    # Load the one-water config
    config_path = Path(__file__).parent / "config_one_water.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    sim = Simulation.from_config(config)
    plotter = SimulationPlotter(sim)
    plotter.animate_simulation()

if __name__ == "__main__":
    main()
