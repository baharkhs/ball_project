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

    # Add water molecules
    center_positions = [
        (0, 0, well_height / 2),
        (0.3, 0.3, well_height / 2)  # Slightly offset to distinguish the second molecule
    ]
    for i, center_position in enumerate(center_positions):
        sim.create_water_molecule(center_position=center_position, velocity=(0.0, 0.0, 0.0), molecule_id=i + 1)

    return sim


def update_animation(frame, sim, ball_plots, path_plots, bond_lines, show_path):
    """
    Updates the animation for each frame.
    """
    sim.update()  # Update simulation state

    for i, ball in enumerate(sim.balls):
        # Update ball positions
        ball_plots[i].set_data([ball.position[0]], [ball.position[1]])
        ball_plots[i].set_3d_properties([ball.position[2]])

        # Update paths if show_path is enabled
        if show_path and path_plots[i] is not None:
            x_path, y_path, z_path = [], [], []
            for segment in ball.get_path_segments():
                x_path += segment["x"]
                y_path += segment["y"]
                z_path += segment["z"]
            path_plots[i].set_data(x_path, y_path)
            path_plots[i].set_3d_properties(z_path)
        elif path_plots[i] is not None:
            path_plots[i].set_data([], [])
            path_plots[i].set_3d_properties([])

    # Update bonds for water molecules
    for i, bond_line in enumerate(bond_lines):
        oxygen = sim.balls[i * 3]  # Oxygen
        hydrogen1 = sim.balls[i * 3 + 1]  # First Hydrogen
        hydrogen2 = sim.balls[i * 3 + 2]  # Second Hydrogen

        bond_line[0].set_data([oxygen.position[0], hydrogen1.position[0]],
                              [oxygen.position[1], hydrogen1.position[1]])
        bond_line[0].set_3d_properties([oxygen.position[2], hydrogen1.position[2]])

        bond_line[1].set_data([oxygen.position[0], hydrogen2.position[0]],
                              [oxygen.position[1], hydrogen2.position[1]])
        bond_line[1].set_3d_properties([oxygen.position[2], hydrogen2.position[2]])

    return ball_plots + [line for bond in bond_lines for line in bond]


def run_simulation(config, show_path=False):
    """
    Runs and visualizes the simulation.

    Args:
        config (dict): Configuration dictionary for the simulation.
        show_path (bool): Flag to enable/disable path visualization.
    """
    sim = setup_simulation(config)

    # Set up the 3D visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-sim.well.radius, sim.well.radius)
    ax.set_ylim(-sim.well.radius, sim.well.radius)
    ax.set_zlim(0, sim.well.height)

    ax.set_title("Simulation of H2O Molecules")
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")

    # Plot the well boundary using the Well class
    sim.well.plot_boundary(ax)

    # Create ball and path plots
    num_balls = len(sim.balls)
    ball_plots = [ax.plot([], [], [], 'o', color=ball.color, markersize=ball.size)[0] for ball in sim.balls]

    # Initialize path plots conditionally
    path_plots = []
    if show_path:
        path_plots = [ax.plot([], [], [], color='blue', linewidth=0.8)[0] for _ in range(num_balls)]
    else:
        path_plots = [None for _ in range(num_balls)]

    # Create bond lines for water molecules
    num_molecules = num_balls // 3
    bond_lines = []
    for _ in range(num_molecules):
        line1, = ax.plot([], [], [], color="gray", linewidth=1)
        line2, = ax.plot([], [], [], color="gray", linewidth=1)
        bond_lines.append([line1, line2])

    # Run the animation
    ani = animation.FuncAnimation(
        fig, update_animation, frames=int(sim.total_time / sim.dt), interval=sim.dt * 1000,
        fargs=(sim, ball_plots, path_plots, bond_lines, show_path)
    )

    plt.show()

    # Plot temperature history
    if hasattr(sim, "temperature_history") and sim.temperature_history:
        plt.figure()
        plt.plot(sim.temperature_history, color="blue", linewidth=1.5)
        plt.title("System Temperature Over Time")
        plt.xlabel("Simulation Step")
        plt.ylabel("Temperature (K)")
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
