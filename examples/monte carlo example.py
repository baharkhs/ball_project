import sys
import os
import json
from ball_simulation.animation import Simulation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def load_simulation_config(config_path="config.json", mode="monte_carlo"):
    """
    Load simulation configurations from a JSON file for the specified mode.

    Args:
        config_path (str): Path to the configuration file.
        mode (str): Mode of simulation ("newtonian" or "monte_carlo").

    Returns:
        dict: Configuration dictionary for the specified mode.
    """
    # Resolve the absolute path relative to the script's directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(project_root, config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    if mode not in config:
        raise KeyError(f"Mode '{mode}' not found in configuration file.")

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

    # Add balls
    for ball_config in config["balls"]:
        sim.add_ball(
            mass=ball_config["mass"],
            initial_position=ball_config["initial_position"],
            initial_velocity=ball_config["initial_velocity"]
        )

    return sim


def update_animation(frame, sim, ball_plots, path_plots):
    """
    Updates the animation for each frame.

    Args:
        frame (int): Current frame number.
        sim (Simulation): The simulation object.
        ball_plots (list): Matplotlib Line3D objects for balls.
        path_plots (list): Matplotlib Line3D objects for paths.

    Returns:
        list: Updated artists for animation.
    """
    sim.update()

    for i, ball in enumerate(sim.balls):
        # Update ball position
        ball_plots[i].set_data([ball.position[0]], [ball.position[1]])
        ball_plots[i].set_3d_properties([ball.position[2]])

        # Update path (exclude PBC jumps)
        x, y, z = [], [], []
        for segment in ball.get_path_segments():
            x.extend(segment["x"])
            y.extend(segment["y"])
            z.extend(segment["z"])
        path_plots[i].set_data(x, y)
        path_plots[i].set_3d_properties(z)

    return ball_plots + path_plots


def visualize_simulation(sim, config):
    """
    Visualize the simulation using Matplotlib animation.

    Args:
        sim (Simulation): The simulation object to visualize.
        config (dict): Configuration dictionary for visual properties of balls.
    """
    # Set up the animation plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-sim.well.radius, sim.well.radius)
    ax.set_ylim(-sim.well.radius, sim.well.radius)
    ax.set_zlim(0, sim.well.height)

    # Draw the well boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    z = np.linspace(0, sim.well.height, 50)
    theta, z = np.meshgrid(theta, z)
    x = sim.well.radius * np.cos(theta)
    y = sim.well.radius * np.sin(theta)
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.3)

    # Initialize ball plots and path plots
    ball_plots = [
        ax.plot([], [], [], "o", color=ball["color"], markersize=ball["size"])[0] for ball in config["balls"]
    ]
    path_plots = [
        ax.plot([], [], [], color=ball["color"], linewidth=0.8)[0] for ball in config["balls"]
    ]

    # Create the animation
    ani = animation.FuncAnimation(
        fig, update_animation, frames=int(sim.total_time / sim.dt), interval=sim.dt * 1000,
        fargs=(sim, ball_plots, path_plots)
    )

    plt.show()


def main():
    """
    Main function to demonstrate running and visualizing a simulation with Monte Carlo movement.
    """
    # Load the configuration
    config_path = "examples/config.json"
    mode = "monte_carlo"  # Ensure we use Monte Carlo settings
    config = load_simulation_config(config_path, mode=mode)

    # Set up the simulation
    sim = setup_simulation(config)

    # Visualize the simulation
    visualize_simulation(sim, config)


if __name__ == "__main__":
    main()
