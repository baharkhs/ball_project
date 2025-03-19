import json
from pathlib import Path
from src.simulation import Simulation
from src.plotting import SimulationPlotter

def main():
    # Load the single-ball config
    config_path = Path(__file__).parent / "config_single_ball.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    # Create simulation from config
    sim = Simulation.from_config(config)

    # Plot/animate
    plotter = SimulationPlotter(sim)
    plotter.animate_simulation()

if __name__ == "__main__":
    main()
