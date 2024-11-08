import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ball_simulation.animation import Simulation
import numpy as np


def generate_random_position(well_radius, well_height):
    """
    Generates a random position inside a cylindrical well.

    Args:
        well_radius (float): The radius of the well in the x-y plane.
        well_height (float): The height of the well along the z-axis.

    Returns:
        np.array: A random position [x, y, z] within the well.
    """
    # Random angle and radius within the circular x-y boundary
    angle = np.random.uniform(0, 2 * np.pi)
    radius = np.random.uniform(0, well_radius)
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    # Random z within the well height
    z = np.random.uniform(0, well_height)
    return np.array([x, y, z])


def example_simulation():
    """
    Sets up and runs a simulation of particle-like behavior in a cylindrical well with elastic bouncing behavior.

    This example simulates two balls using Newtonian movement with elastic boundaries.
    Periodic boundary conditions (PBC) are applied only in the vertical (z) direction.
    """

    # Initialize the simulation with specific well dimensions and time settings
    well_radius = 0.6
    well_height = 1.0
    sim = Simulation(well_radius=well_radius, well_height=well_height, total_time=15.0, dt=0.02)

    # Define properties for each ball with random initial positions
    balls_config = [
        {
            "mass": 2.0,
            "initial_position": generate_random_position(well_radius, well_height),
            "initial_velocity": [0.1, -0.05, 0.05],  # Adjust initial velocity if desired
            "color": "red",
            "size": 10
        },
        {
            "mass": 1.2,
            "initial_position": generate_random_position(well_radius, well_height),
            "initial_velocity": [-0.08, 0.07, 0.04],  # Adjust initial velocity if desired
            "color": "blue",
            "size": 7
        }
    ]

    # Add each ball to the simulation
    for config in balls_config:
        sim.add_ball(
            mass=config["mass"],
            initial_position=config["initial_position"],
            initial_velocity=config["initial_velocity"]
        )

    # Set up the animation plot
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
    ball_plots = [ax.plot([], [], [], 'o', color=config["color"], markersize=config["size"])[0] for config in
                  balls_config]
    path_plots = [ax.plot([], [], [], color=config["color"], linewidth=0.8)[0] for config in balls_config]

    def update_animation(frame):
        # Update the simulation for each frame
        sim.update()
        for i, ball in enumerate(sim.balls):
            # Update ball plot
            ball_plots[i].set_data([ball.position[0]], [ball.position[1]])
            ball_plots[i].set_3d_properties([ball.position[2]])

            # Update path plot to match ball's movement
            path_plots[i].set_data(ball.path_x, ball.path_y)
            path_plots[i].set_3d_properties(ball.path_z)

        return ball_plots + path_plots

    # Create the animation
    ani = animation.FuncAnimation(fig, update_animation, frames=int(sim.total_time / sim.dt), interval=sim.dt * 1000)
    plt.show()


if __name__ == "__main__":
    example_simulation()
