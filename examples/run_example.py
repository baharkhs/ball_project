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

    # Add balls to the simulation
    for ball_config in config["balls"]:
        sim.add_ball(
            mass=ball_config["mass"],
            initial_position=ball_config["initial_position"],
            initial_velocity=ball_config["initial_velocity"],
            species=ball_config["species"],
            molecule_id=ball_config["molecule_id"]
        )

    return sim


def update_animation(frame, sim, ball_plots, path_plots, show_path):
    """
    Updates the animation for each frame.

    Args:
        frame (int): The current frame of the animation.
        sim (Simulation): The simulation object to update.
        ball_plots (list): List of ball plots.
        path_plots (list): List of path plots.
        show_path (bool): Flag to enable/disable path visualization.

    Returns:
        list: Updated plots for animation.
    """
    sim.update()  # Update simulation state

    for i, ball in enumerate(sim.balls):
        # Update ball positions
        ball_plots[i].set_data([ball.position[0]], [ball.position[1]])
        ball_plots[i].set_3d_properties([ball.position[2]])

        # Combine and update paths if enabled
        if show_path:
            x_path, y_path, z_path = [], [], []
            for segment in ball.get_path_segments():
                x_path += segment["x"]
                y_path += segment["y"]
                z_path += segment["z"]
            path_plots[i].set_data(x_path, y_path)
            path_plots[i].set_3d_properties(z_path)

    return ball_plots + path_plots


def run_simulation(config, show_path=True):
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

    # Initialize ball and path plots
    ball_configs = config["balls"]
    ball_plots = [ax.plot([], [], [], 'o', color=ball["color"], markersize=ball["size"])[0] for ball in ball_configs]
    path_plots = [ax.plot([], [], [], color=ball["color"], linewidth=0.8)[0] for ball in ball_configs]

    # Create the animation
    ani = animation.FuncAnimation(
        fig, update_animation, frames=int(sim.total_time / sim.dt), interval=sim.dt * 1000,
        fargs=(sim, ball_plots, path_plots, show_path)
    )

    plt.show()

    # Plot temperature history
    if hasattr(sim, "temperature_history"):
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
    show_path = True  # Set to False to disable paths
    run_simulation(config, show_path=show_path)


if __name__ == "__main__":
    main()
