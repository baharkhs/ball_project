import matplotlib.pyplot as plt
import sys
import matplotlib.animation as animation
from ball_simulation.animation import Simulation
from ball_simulation.input_output import load_config  # Load from input_output
import numpy as np


def check_python_version():
    """Prints the current Python version."""
    version = sys.version_info
    print(f"Running Python version: {version.major}.{version.minor}.{version.micro}")


def print_units():
    """Prints the units used in the simulation."""
    print("Units used in the simulation:")
    print(" - Well radius and height: Ångström (10⁻¹⁰ m)")
    print(" - Ball mass: Atomic Mass Unit (amu)")
    print(" - Velocity: Ångström/picosecond (Å/ps)")
    print(" - Temperature: Kelvin (K)")
    print(" - Time step: Picoseconds (ps)")
    print(" - Total simulation time: Picoseconds (ps)")


def example_simulation():
    """
    Sets up and runs a simulation using configurations from a JSON file.
    """
    # Load configurations
    config = load_config("config.json")

    # Extract simulation parameters
    well_radius = config["well"]["radius"]
    well_height = config["well"]["height"]
    total_time = config["simulation"]["total_time"]
    dt = config["simulation"]["dt"]
    movement_type = config["simulation"]["movement_type"]

    # Set up the simulation
    sim = Simulation(well_radius=well_radius, well_height=well_height, total_time=total_time, dt=dt)
    sim.set_movement_type(movement_type)

    # Add balls to the simulation using configurations
    for ball_config in config["balls"]:
        sim.add_ball(
            mass=ball_config["mass"],
            initial_position=ball_config["initial_position"],
            initial_velocity=ball_config["initial_velocity"]
        )

    # Set up the animation plot (same as before)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-sim.well.radius, sim.well.radius)
    ax.set_ylim(-sim.well.radius, sim.well.radius)
    ax.set_zlim(0, sim.well.height)

    # Draw the well boundary as a cylinder
    theta = np.linspace(0, 2 * np.pi, 100)
    z = np.linspace(0, sim.well.height, 50)
    theta, z = np.meshgrid(theta, z)
    x = sim.well.radius * np.cos(theta)
    y = sim.well.radius * np.sin(theta)
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.3)

    # Initialize plot elements for each ball
    ball_plots = [ax.plot([], [], [], 'o', color=ball_config["color"], markersize=ball_config["size"])[0]
                  for ball_config in config["balls"]]
    path_plots = [ax.plot([], [], [], color=ball_config["color"], linewidth=0.8)[0]
                  for ball_config in config["balls"]]

    def update_animation(frame):
        sim.update()
        for i, ball in enumerate(sim.balls):
            ball_plots[i].set_data([ball.position[0]], [ball.position[1]])
            ball_plots[i].set_3d_properties([ball.position[2]])

            # Plot segmented paths without vertical lines
            path_plots[i].set_data([], [])
            path_plots[i].set_3d_properties([])
            for segment in ball.get_path_segments():
                path_plots[i].set_data(segment["x"], segment["y"])
                path_plots[i].set_3d_properties(segment["z"])

        return ball_plots + path_plots

    # Create the animation
    ani = animation.FuncAnimation(fig, update_animation, frames=int(sim.total_time / sim.dt), interval=sim.dt * 1000)
    plt.show()


if __name__ == "__main__":
    check_python_version()
    print_units()
    example_simulation()
