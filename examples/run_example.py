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

    # Initialize the simulation object
    sim = Simulation(well_radius=well_radius, well_height=well_height, total_time=total_time, dt=dt)
    sim.set_movement_type(movement_type)

    # Add balls from config
    for ball_config in config["balls"]:
        sim.add_ball(
            mass=ball_config["mass"],
            initial_position=ball_config["initial_position"],
            initial_velocity=ball_config["initial_velocity"],
            species=ball_config.get("species", "O"),
            molecule_id=ball_config.get("molecule_id", None)
        )

    # Stabilize initial conditions by rescaling to a reasonable temperature (e.g., 300 K)
    sim.rescale_temperatures(300.0, decay_factor=1.0)  # Immediate rescale to a stable temperature

    return sim

def update_animation(frame, sim, ball_plots, path_plots, ball_configs, times, temperatures):
    """
    Update function for Matplotlib's FuncAnimation.

    We:
    - Advance the simulation by one step.
    - Record time and temperature.
    - Update ball positions and paths in the 3D plot.
    """
    sim.update()
    current_time = frame * sim.dt
    times.append(current_time)
    temperatures.append(sim.average_temperature)

    # Print diagnostics
    print(
        f"Frame {frame}: Time = {current_time:.2f} fs, Avg Temp = {sim.average_temperature:.2f} K, Total Energy = {sim.total_energy:.2f}")

    # Update ball positions
    for i, ball in enumerate(sim.balls):
        ball_plots[i].set_data([ball.position[0]], [ball.position[1]])
        ball_plots[i].set_3d_properties([ball.position[2]])

        # Update paths
        path_data = ball.get_path_segments()
        if path_data:
            latest_segment = path_data[-1]
            if latest_segment["x"]:
                path_plots[i].set_data(latest_segment["x"], latest_segment["y"])
                path_plots[i].set_3d_properties(latest_segment["z"])

    return ball_plots + path_plots

def visualize_simulation(sim, config, target_temperature):
    """
    Visualize the simulation using a 3D animation.

    We:
    - Run the simulation step-by-step via FuncAnimation.
    - Record temperature and time data.
    - After the figure closes, plot the temperature vs time.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-sim.well.radius, sim.well.radius)
    ax.set_ylim(-sim.well.radius, sim.well.radius)
    ax.set_zlim(0, sim.well.height)

    ax.set_title("Simulation of Ball Movements")
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")

    # Draw cylindrical well boundary
    theta = np.linspace(0, 2 * np.pi, 150)
    z = np.linspace(0, sim.well.height, 75)
    theta, z = np.meshgrid(theta, z)
    x = sim.well.radius * np.cos(theta)
    y = sim.well.radius * np.sin(theta)
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.3, linestyle="dotted", linewidth=0.5)

    ball_configs = config["balls"]
    ball_plots = [ax.plot([], [], [], 'o', color=ball["color"], markersize=ball["size"])[0] for ball in ball_configs]
    path_plots = [ax.plot([], [], [], color=ball["color"], linewidth=0.8)[0] for ball in ball_configs]

    # Create a legend for species
    unique_species = set(ball["species"] for ball in ball_configs)
    for species in unique_species:
        ax.scatter([], [], [], label=f"Species: {species}", s=50)
    ax.legend(loc="upper right", title="Legend")

    # Prepare data storage for temperature vs time
    times = []
    temperatures = []
    total_steps = int(sim.total_time / sim.dt)

    # Create animation that runs the simulation step-by-step
    ani = animation.FuncAnimation(
        fig,
        update_animation,
        frames=total_steps,
        interval=sim.dt * 1000,
        fargs=(sim, ball_plots, path_plots, ball_configs, times, temperatures),
        blit=False,
        repeat=False
    )

    def on_close(event):
        # After closing animation window, plot temperature vs time
        plt.figure()
        plt.plot(times, temperatures, label="Average Temperature", color="blue")
        if target_temperature is not None:
            plt.axhline(y=target_temperature, color="red", linestyle="--", label="Target Temperature")
        plt.xlabel("Time (fs)")
        plt.ylabel("Temperature (K)")
        plt.title("Temperature vs Time")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    fig.canvas.mpl_connect('close_event', on_close)

    plt.show()

def main():
    mode = "newtonian"  # change to "monte_carlo" if desired
    config_path = "examples/config.json"

    # Load configuration (do not change related lines)
    config = load_simulation_config(config_path, mode=mode)

    # Setup simulation
    sim = setup_simulation(config)

    target_temperature = 300.0

    # Visualize simulation and record temperature
    visualize_simulation(sim, config, target_temperature)

if __name__ == "__main__":
    main()
