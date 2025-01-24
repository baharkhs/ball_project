
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Import from your local modules
from ball_simulation.simulation import Simulation
from ball_simulation.plotting import SimulationPlotter

def load_simulation_config(config_path="config.json", mode="newtonian"):
    """
    Load simulation configurations from a JSON file.
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
    sim = Simulation(
        well_radius=well_radius,
        well_height=well_height,
        total_time=total_time,
        dt=dt
    )
    sim.set_movement_type(movement_type)

    # Add water molecules (H2O)
    oxygen_molecules = config["particles"].get("oxygen_molecules", [])
    for molecule in oxygen_molecules:
        center_position = molecule["center_position"]
        molecule_id = molecule["molecule_id"]
        sim.create_water_molecule(center_position=center_position, molecule_id=molecule_id)

    # Add custom particles if needed
    custom_particles = config["particles"].get("custom_particles", [])
    for particle in custom_particles:
        species = particle["species"]
        position = particle["position"]
        velocity = particle["velocity"]
        mass = particle["mass"]
        sim.add_ball(
            mass=mass,
            initial_position=position,
            initial_velocity=velocity,
            species=species
        )

    return sim


def update_animation(frame, sim, ball_plots, temp_line, ax_temp):
    """
    Updates the animation for each frame.
    """
    # Advance simulation by one step
    sim.update()

    # Update ball positions in 3D
    for i, ball in enumerate(sim.balls):
        ball_plots[i].set_data([ball.position[0]], [ball.position[1]])
        ball_plots[i].set_3d_properties([ball.position[2]])

    # Update temperature plot
    steps = np.arange(len(sim.temperature_history))
    temp_line.set_data(steps, sim.temperature_history)

    # Rescale the temperature axes if needed
    ax_temp.set_xlim(0, max(1, len(steps)))
    if len(sim.temperature_history) > 0:
        max_temp = max(sim.temperature_history)
        ax_temp.set_ylim(0, max_temp + 50)

    return ball_plots + [temp_line]


def run_simulation(config):
    """
    Runs and visualizes the simulation with a real-time 3D animation + temperature plot,
    then plots the potential energy data after completion.
    """
    # 1) Set up the Simulation object
    sim = setup_simulation(config)

    # 2) Figure and subplots
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])

    ax_sim = fig.add_subplot(gs[0], projection='3d')
    ax_temp = fig.add_subplot(gs[1])

    # 3) Configure the 3D simulation axes
    ax_sim.set_xlim(-sim.well.radius, sim.well.radius)
    ax_sim.set_ylim(-sim.well.radius, sim.well.radius)
    ax_sim.set_zlim(0, sim.well.height)
    ax_sim.set_title("Simulation of H2O Molecules")
    ax_sim.set_xlabel("X (Å)")
    ax_sim.set_ylabel("Y (Å)")
    ax_sim.set_zlabel("Z (Å)")

    # Draw the cylindrical boundary
    sim.well.plot_boundary(ax_sim)

    # 4) Configure the temperature subplot
    ax_temp.set_title("System Temperature Over Time")
    ax_temp.set_xlabel("Simulation Step")
    ax_temp.set_ylabel("Temperature (K)")
    temp_line, = ax_temp.plot([], [], color="blue", label="Temperature")
    ax_temp.legend()
    ax_temp.grid(True)

    # 5) Create ball plots
    ball_plots = [
        ax_sim.plot([], [], [], 'o', color=b.color, markersize=b.size)[0]
        for b in sim.balls
    ]

    # 6) Define update function for animation
    def update_frame(frame):
        return update_animation(frame, sim, ball_plots, temp_line, ax_temp)

    # 7) Create the animation
    frames = int(sim.total_time / sim.dt)
    ani = animation.FuncAnimation(
        fig, update_frame,
        frames=frames,
        interval=sim.dt * 1000,  # in milliseconds
        blit=False
    )

    # 8) Show the real-time animation
    plt.show()

    # 9) After the simulation finishes, plot potential energy
    if sim.potential_energy_data:
        # Potential energy data is stored as (distance, potential) in sim.potential_energy_data
        distances, potential_energies = zip(*sim.potential_energy_data)
        plt.figure(figsize=(8, 6))
        plt.scatter(distances, potential_energies, color="blue", alpha=0.6)
        plt.title("Potential Energy vs Distance")
        plt.xlabel("Distance (Å)")
        plt.ylabel("Potential Energy (eV)")
        plt.grid(True)
        plt.show()
    else:
        print("No potential energy data was collected in sim.potential_energy_data.")


    plotter = SimulationPlotter(sim)
    plotter.plot_all()


def main():
    """
    Main function to run the simulation.
    """
    mode = "newtonian"
    config_path = "examples/config.json"
    config = load_simulation_config(config_path, mode=mode)
    run_simulation(config)

if __name__ == "__main__":
    main()
