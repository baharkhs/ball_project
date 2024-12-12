import os
import json
import sys
from ball_simulation.animation import Simulation
from ball_simulation.input_output import load_config
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from ball_simulation.animation import Simulation
from ball_simulation.animation import Ball
from ball_simulation.animation import Well  # Import the Well class


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
    # Do NOT change this line related to reading the json file:
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
    """
    # Extract simulation parameters
    well_radius = config["well"]["radius"]
    well_height = config["well"]["height"]
    total_time = config["simulation"]["total_time"]
    dt = config["simulation"]["dt"]
    movement_type = config["simulation"]["movement_type"]

    # Initialize the Well object
    well = Well(radius=well_radius, height=well_height)

    # Initialize the Simulation object
    sim = Simulation(
        well_radius=well_radius,
        well_height=well_height,
        total_time=total_time,
        dt=dt,
        movement_type=movement_type
    )
    sim.well = well  # Attach the Well to the Simulation

    # Add Balls to the Simulation
    for ball_config in config["balls"]:
        ball = Ball(
            position=ball_config["initial_position"],
            velocity=ball_config["initial_velocity"],
            mass=ball_config["mass"],
            radius=ball_config["radius"],
            attraction_coeff=ball_config.get("attraction_coeff", 0.0),
            repulsion_coeff=ball_config.get("repulsion_coeff", 0.0)
        )
        sim.add_ball(ball)

    # Stabilize the initial conditions
    sim.apply_thermostat(target_temperature=300.0, method="nose_hoover")

    return sim

def update_animation(frame, sim, ball_plots, path_plots, times, temperatures):
    """
    Update the simulation and visualization for each frame.
    """
    sim.update()
    current_time = frame * sim.dt
    times.append(current_time)
    temperatures.append(sim.calculate_average_temperature())

    # Print diagnostics
    print(
        f"Frame {frame}: Time = {current_time:.2f} fs, Avg Temp = {temperatures[-1]:.2f} K, Total Energy = {sim.calculate_total_energy():.2f}"
    )

    # Update ball positions
    for i, ball in enumerate(sim.balls):
        ball_plots[i].set_data([ball.position[0]], [ball.position[1]])
        ball_plots[i].set_3d_properties([ball.position[2]])

        # Update paths
        path_segments = ball.get_all_paths()
        if path_segments:
            latest_segment = path_segments[-1]
            path_plots[i].set_data(latest_segment["x"], latest_segment["y"])
            path_plots[i].set_3d_properties(latest_segment["z"])

    return ball_plots + path_plots

def visualize_simulation(sim, config):
    """
    Visualize the simulation using Matplotlib's 3D animation.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-sim.well_radius, sim.well_radius)
    ax.set_ylim(-sim.well_radius, sim.well_radius)
    ax.set_zlim(0, sim.well_height)
    ax.set_title("Simulation of Ball Movements")
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")

    # Initialize balls and paths
    ball_configs = config["balls"]
    ball_plots = [
        ax.plot([], [], [], "o", color=ball["color"], markersize=ball["size"])[0] for ball in ball_configs
    ]
    path_plots = [
        ax.plot([], [], [], color=ball["color"], linewidth=0.8)[0] for ball in ball_configs
    ]

    # Data for temperature vs time
    times = []
    temperatures = []

    # Total steps
    total_steps = int(sim.total_time / sim.dt)

    # Animation
    ani = animation.FuncAnimation(
        fig,
        update_animation,
        frames=total_steps,
        interval=sim.dt * 1000,
        fargs=(sim, ball_plots, path_plots, times, temperatures),
        blit=False,
        repeat=False
    )

    def on_close(event):
        plt.figure()
        plt.plot(times, temperatures, label="Avg Temperature", color="blue")
        plt.axhline(y=300.0, color="red", linestyle="--", label="Target Temp (300 K)")
        plt.xlabel("Time (fs)")
        plt.ylabel("Temperature (K)")
        plt.title("Temperature vs Time")
        plt.legend()
        plt.grid()
        plt.show()

    fig.canvas.mpl_connect("close_event", on_close)
    plt.show()

def main():
    mode = "newtonian"  # Set simulation mode
    config_path = "examples/config.json"  # Path to the configuration file

    # Load configuration
    config = load_simulation_config(config_path, mode=mode)

    # Set up the simulation
    sim = setup_simulation(config)

    # Visualize the simulation
    visualize_simulation(sim, config)

if __name__ == "__main__":
    main()