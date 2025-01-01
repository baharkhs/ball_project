import sys
import os
import json
from ball_simulation.animation import Simulation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Debugging: Print the current working directory
print("Current working directory:", os.getcwd())


def load_simulation_config(config_path="config.json", mode="newtonian"):
    """
    Load simulation configurations from a JSON file.

    Args:
        config_path (str): Path to the configuration file.
        mode (str): Simulation mode to load ("newtonian" or "monte_carlo").

    Returns:
        dict: Configuration dictionary for the specified mode.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(project_root, config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    if mode not in config:
        raise ValueError(f"Mode '{mode}' not found in the configuration file.")

    return config[mode]


def setup_simulation(config):
    """
    Set up the simulation using the provided configuration.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        Simulation: Initialized simulation object.
    """
    # Extract simulation parameters
    well_radius = config["well"]["radius"]
    well_height = config["well"]["height"]
    total_time = config["simulation"]["total_time"]
    dt = config["simulation"]["dt"]
    movement_type = config["simulation"]["movement_type"]

    # Initialize the simulation
    sim = Simulation(well_radius=well_radius, well_height=well_height, total_time=total_time, dt=dt)
    sim.set_movement_type(movement_type)

    # Add water molecules or balls
    center_positions = [
        (0, 0, well_height / 2),  # Center of the well
        (0.3, 0.3, well_height / 2),  # Offset slightly
        (-0.3, -0.3, well_height / 2)  # Another position for testing
    ]

    for i, center_position in enumerate(center_positions):
        sim.create_water_molecule(center_position=center_position, velocity=(0.0, 0.0, 0.0), molecule_id=i + 1)

    return sim


def update_frame(frame, sim, ball_plots, temp_line, ax_temp):
    """
    Update function for animation.
    """
    sim.update()  # Update simulation state

    # Update ball positions
    for i, ball in enumerate(sim.balls):
        ball_plots[i].set_data([ball.position[0]], [ball.position[1]])
        ball_plots[i].set_3d_properties([ball.position[2]])

    # Update temperature plot
    if len(sim.temperature_history) > 0:
        temp_line.set_data(range(len(sim.temperature_history)), sim.temperature_history)
        ax_temp.set_xlim(0, len(sim.temperature_history))
        max_temp = max(sim.temperature_history) if max(sim.temperature_history) > 0 else 1
        ax_temp.set_ylim(0, max_temp * 1.1)

    return ball_plots + [temp_line]


def run_simulation(config, show_path=True):
    """
    Runs and visualizes the simulation.

    Args:
        config (dict): Configuration dictionary for the simulation.
        show_path (bool): Flag to enable/disable path visualization.
    """
    sim = setup_simulation(config)

    if not sim.balls:
        raise ValueError("No balls present in the simulation. Ensure balls are added during setup.")

    # Set up the 3D visualization
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    ax_sim = fig.add_subplot(gs[0], projection='3d')
    ax_temp = fig.add_subplot(gs[1])

    # Set up simulation plot
    ax_sim.set_xlim(-sim.well.radius, sim.well.radius)
    ax_sim.set_ylim(-sim.well.radius, sim.well.radius)
    ax_sim.set_zlim(0, sim.well.height)
    ax_sim.set_title("Simulation of H2O Molecules")
    ax_sim.set_xlabel("X (Å)")
    ax_sim.set_ylabel("Y (Å)")
    ax_sim.set_zlabel("Z (Å)")
    sim.well.plot_boundary(ax_sim)

    # Set up temperature plot
    ax_temp.set_title("System Temperature Over Time")
    ax_temp.set_xlabel("Simulation Step")
    ax_temp.set_ylabel("Temperature (K)")
    temp_line, = ax_temp.plot([], [], color="blue", label="Temperature")
    ax_temp.legend()
    ax_temp.grid(True)

    # Create ball plots
    ball_plots = [ax_sim.plot([], [], [], 'o', color=getattr(ball, 'color', 'blue'),
                              markersize=getattr(ball, 'size', 6))[0] for ball in sim.balls]

    ani = animation.FuncAnimation(
        fig, update_frame, frames=int(sim.total_time / sim.dt), interval=sim.dt * 1000,
        fargs=(sim, ball_plots, temp_line, ax_temp)
    )

    plt.show()

    # Plot potential energy after the simulation
    if sim.potential_energy_data:
        distances, potential_energies = zip(*sim.potential_energy_data)
        plt.figure(figsize=(8, 6))
        plt.scatter(distances, potential_energies, color="blue", alpha=0.6, label="Simulation Data")
        plt.title("Potential Energy vs Distance")
        plt.xlabel("Distance (Å)")
        plt.ylabel("Potential Energy (eV)")
        plt.legend()
        plt.grid(True)
        plt.show()


def main():
    """
    Main function to demonstrate running and visualizing a simulation.
    """
    mode = "newtonian"  # Change this to "monte_carlo" for Monte Carlo simulation
    config_path = "examples/config.json"

    # Load configuration and run simulation
    config = load_simulation_config(config_path, mode=mode)

    # Toggle path visualization ON/OFF
    show_path = False  # Set to True to enable paths
    run_simulation(config, show_path=show_path)


if __name__ == "__main__":
    main()
