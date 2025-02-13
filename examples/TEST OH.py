import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ball_simulation.simulation import Simulation

# Create a simulation instance with a well large enough for visualization.
sim = Simulation(well_radius=5.0, well_height=5.0, total_time=5.0, dt=0.001)

# Define a nonzero initial velocity in all directions.
initial_velocity = [0.2, 0.2, 0.2]

# Create one water molecule using create_water_molecule.
# This creates one oxygen and two hydrogens, all sharing molecule_id "H2O".
sim.create_water_molecule(center_position=[0.0, 0.0, 1.0],
                          velocity=initial_velocity,
                          molecule_id="H2O")

# Set up the figure and 3D axis for animation.
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-sim.well.radius, sim.well.radius)
ax.set_ylim(-sim.well.radius, sim.well.radius)
ax.set_zlim(0, sim.well.height)
ax.set_title("Minimal Water Molecule Simulation")
ax.set_xlabel("X (Å)")
ax.set_ylabel("Y (Å)")
ax.set_zlabel("Z (Å)")

# Optionally, draw the well boundary.
sim.well.plot_boundary(ax)

# Create scatter plots for each ball.
# We assume the molecule "H2O" exists.
mol = sim.molecules["H2O"]
scatter_O = ax.scatter(sim.balls[mol["O"]].position[0],
                       sim.balls[mol["O"]].position[1],
                       sim.balls[mol["O"]].position[2],
                       c="red", s=150, label="O")
scatter_H1 = ax.scatter(sim.balls[mol["H1"]].position[0],
                        sim.balls[mol["H1"]].position[1],
                        sim.balls[mol["H1"]].position[2],
                        c="blue", s=50, label="H1")
scatter_H2 = ax.scatter(sim.balls[mol["H2"]].position[0],
                        sim.balls[mol["H2"]].position[1],
                        sim.balls[mol["H2"]].position[2],
                        c="blue", s=50, label="H2")

# Create initial lines (bonds) connecting O to H1 and O to H2.
bond_line_H1, = ax.plot([sim.balls[mol["O"]].position[0], sim.balls[mol["H1"]].position[0]],
                        [sim.balls[mol["O"]].position[1], sim.balls[mol["H1"]].position[1]],
                        [sim.balls[mol["O"]].position[2], sim.balls[mol["H1"]].position[2]],
                        color="green", linewidth=2, label="Bond O-H1")
bond_line_H2, = ax.plot([sim.balls[mol["O"]].position[0], sim.balls[mol["H2"]].position[0]],
                        [sim.balls[mol["O"]].position[1], sim.balls[mol["H2"]].position[1]],
                        [sim.balls[mol["O"]].position[2], sim.balls[mol["H2"]].position[2]],
                        color="green", linewidth=2, label="Bond O-H2")


def update_animation(frame):
    # Advance the simulation one time step.
    sim.update()

    # Update scatter plots for the oxygen and hydrogens.
    O_pos = sim.balls[mol["O"]].position
    H1_pos = sim.balls[mol["H1"]].position
    H2_pos = sim.balls[mol["H2"]].position
    scatter_O._offsets3d = ([O_pos[0]], [O_pos[1]], [O_pos[2]])
    scatter_H1._offsets3d = ([H1_pos[0]], [H1_pos[1]], [H1_pos[2]])
    scatter_H2._offsets3d = ([H2_pos[0]], [H2_pos[1]], [H2_pos[2]])

    # Update bond lines.
    bond_line_H1.set_data([O_pos[0], H1_pos[0]], [O_pos[1], H1_pos[1]])
    bond_line_H1.set_3d_properties([O_pos[2], H1_pos[2]])

    bond_line_H2.set_data([O_pos[0], H2_pos[0]], [O_pos[1], H2_pos[1]])
    bond_line_H2.set_3d_properties([O_pos[2], H2_pos[2]])

    # Print current O-H bond lengths.
    d1 = np.linalg.norm(O_pos - H1_pos)
    d2 = np.linalg.norm(O_pos - H2_pos)
    print(f"Step {sim.current_step}: O-H1 = {d1:.3f} Å, O-H2 = {d2:.3f} Å")

    return scatter_O, scatter_H1, scatter_H2, bond_line_H1, bond_line_H2


frames = int(sim.total_time / sim.dt)
ani = animation.FuncAnimation(fig, update_animation, frames=frames, interval=sim.dt * 1000, blit=False)
plt.show()

# After simulation, print final bond lengths.
O_pos = sim.balls[mol["O"]].position
H1_pos = sim.balls[mol["H1"]].position
H2_pos = sim.balls[mol["H2"]].position
d1 = np.linalg.norm(O_pos - H1_pos)
d2 = np.linalg.norm(O_pos - H2_pos)
print(f"Final O-H1 bond length: {d1:.3f} Å")
print(f"Final O-H2 bond length: {d2:.3f} Å")
