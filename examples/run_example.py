import sys
import os
import json
from ball_simulation.animation import Simulation
from ball_simulation.input_output import load_config
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
    # Resolve the absolute path relative to the script's directory
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
            initial_velocity=ball_config["initial_velocity"]
        )

    return sim


def update_animation(frame, sim, ball_plots, path_plots, ball_configs):
    """
    Updates the animation for each frame.

    Args:
        frame (int): Current frame number (not used but required by FuncAnimation).
        sim (Simulation): The simulation object to update.
        ball_plots (list): List of Matplotlib Line3D objects for each ball.
        path_plots (list): List of Matplotlib Line3D path plots for each ball.
        ball_configs (list): Configuration for each ball (e.g., color, size).

    Returns:
        list: Updated Line3D objects for animation.
    """
    sim.update()

    for i, ball in enumerate(sim.balls):
        # Update ball positions
        ball_plots[i].set_data([ball.position[0]], [ball.position[1]])
        ball_plots[i].set_3d_properties([ball.position[2]])
        ball_plots[i].set_color(ball_configs[i]["color"])
        ball_plots[i].set_markersize(ball_configs[i]["size"])

        # Update paths
        x_path, y_path, z_path = [], [], []
        for segment in ball.get_path_segments():
            x_path += segment["x"]
            y_path += segment["y"]
            z_path += segment["z"]
        path_plots[i].set_data(x_path, y_path)
        path_plots[i].set_3d_properties(z_path)

    return ball_plots + path_plots


def visualize_simulation(sim, config):
    """
    Visualize the simulation using Matplotlib animation.

    Args:
        sim (Simulation): The simulation object to visualize.
        config (dict): Configuration dictionary.
    """
    # Set up the animation plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-sim.well.radius, sim.well.radius)
    ax.set_ylim(-sim.well.radius, sim.well.radius)
    ax.set_zlim(0, sim.well.height)

    ax.set_title("Simulation of Ball Movements")
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")

    # Draw the well boundary as a cylinder
    theta = np.linspace(0, 2 * np.pi, 100)
    z = np.linspace(0, sim.well.height, 50)
    theta, z = np.meshgrid(theta, z)
    x = sim.well.radius * np.cos(theta)
    y = sim.well.radius * np.sin(theta)
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.3)

    # Initialize ball and path plots
    ball_configs = config["balls"]
    ball_plots = [ax.plot([], [], [], 'o', color=ball["color"], markersize=ball["size"])[0] for ball in ball_configs]
    path_plots = [ax.plot([], [], [], color=ball["color"], linewidth=0.8)[0] for ball in ball_configs]

    def update_animation(frame, sim, ball_plots, path_plots):
        """Updates the animation for each frame."""
        sim.update()
        for i, ball in enumerate(sim.balls):
            # Update ball position
            ball_plots[i].set_data([ball.position[0]], [ball.position[1]])
            ball_plots[i].set_3d_properties([ball.position[2]])

            # Update path
            path_data = ball.get_path_segments()
            for segment in path_data:
                path_plots[i].set_data(segment["x"], segment["y"])
                path_plots[i].set_3d_properties(segment["z"])

        return ball_plots + path_plots

    # Create the animation
    ani = animation.FuncAnimation(
        fig, update_animation, frames=int(sim.total_time / sim.dt), interval=sim.dt * 1000,
        fargs=(sim, ball_plots, path_plots)
    )

    plt.show()



def main():
    """
    Main function to demonstrate running and visualizing a simulation.
    """
    # Determine the simulation mode
    mode = "newtonian"  # Change this to "monte_carlo" to test Monte Carlo simulation

    # Load the configuration for the selected mode
    config_path = "examples/config.json"
    config = load_simulation_config(config_path, mode=mode)

    # Set up the simulation
    sim = setup_simulation(config)

    # Visualize the simulation
    visualize_simulation(sim, config)

if __name__ == "__main__":
    main()