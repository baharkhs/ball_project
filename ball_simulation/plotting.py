import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


class SimulationPlotter:
    def __init__(self, simulation):
        self.sim = simulation

    def animate_simulation(self):
        """Sets up and runs the 3D animation for the simulation, including bond lines and distance printing."""
        fig = plt.figure(figsize=(8, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
        ax_sim = fig.add_subplot(gs[0], projection='3d')
        ax_temp = fig.add_subplot(gs[1])

        # Set axes limits using well dimensions.
        ax_sim.set_xlim(-self.sim.well.radius, self.sim.well.radius)
        ax_sim.set_ylim(-self.sim.well.radius, self.sim.well.radius)
        ax_sim.set_zlim(0, self.sim.well.height)
        ax_sim.set_title("Minimal Water Molecule Simulation")
        ax_sim.set_xlabel("X (Å)")
        ax_sim.set_ylabel("Y (Å)")
        ax_sim.set_zlabel("Z (Å)")
        self.sim.well.plot_boundary(ax_sim)

        # Configure the temperature subplot.
        ax_temp.set_title("System Temperature Over Time")
        ax_temp.set_xlabel("Step")
        ax_temp.set_ylabel("Temperature (K)")
        temp_line, = ax_temp.plot([], [], color="blue")
        ax_temp.grid(True)

        # Create scatter plots for each ball.
        ball_plots = [ax_sim.plot([], [], [], 'o', color=b.color, markersize=b.size)[0]
                      for b in self.sim.balls]

        # Create bond lines for each water molecule.
        # For each molecule (with molecule_id) create two lines: one for O–H1 and one for O–H2.
        bond_lines = {}
        for molecule_id, indices in self.sim.molecules.items():
            O = self.sim.balls[indices["O"]].position
            H1 = self.sim.balls[indices["H1"]].position
            H2 = self.sim.balls[indices["H2"]].position
            line1, = ax_sim.plot([O[0], H1[0]], [O[1], H1[1]], [O[2], H1[2]],
                                 color="green", linewidth=2)
            line2, = ax_sim.plot([O[0], H2[0]], [O[1], H2[1]], [O[2], H2[2]],
                                 color="green", linewidth=2)
            bond_lines[molecule_id] = (line1, line2)

        def update_frame(frame):
            # Advance simulation one time step.
            self.sim.update()
            # Update scatter plots.
            for i, ball in enumerate(self.sim.balls):
                ball_plots[i].set_data([ball.position[0]], [ball.position[1]])
                ball_plots[i].set_3d_properties([ball.position[2]])
            # Update bond lines.
            for molecule_id, indices in self.sim.molecules.items():
                O = self.sim.balls[indices["O"]].position
                H1 = self.sim.balls[indices["H1"]].position
                H2 = self.sim.balls[indices["H2"]].position
                line1, line2 = bond_lines[molecule_id]
                line1.set_data([O[0], H1[0]], [O[1], H1[1]])
                line1.set_3d_properties([O[2], H1[2]])
                line2.set_data([O[0], H2[0]], [O[1], H2[1]])
                line2.set_3d_properties([O[2], H2[2]])
            # Print current bond lengths.
            for molecule_id, indices in self.sim.molecules.items():
                O = self.sim.balls[indices["O"]].position
                H1 = self.sim.balls[indices["H1"]].position
                H2 = self.sim.balls[indices["H2"]].position
                d1 = np.linalg.norm(O - H1)
                d2 = np.linalg.norm(O - H2)
                print(f"Step {self.sim.current_step}: O-H1 = {d1:.3f} Å, O-H2 = {d2:.3f} Å")
            # Update temperature plot.
            steps = np.arange(len(self.sim.temperature_history))
            temp_line.set_data(steps, self.sim.temperature_history)
            ax_temp.set_xlim(0, max(1, len(steps)))
            if self.sim.temperature_history:
                ax_temp.set_ylim(0, max(self.sim.temperature_history) + 50)
            return ball_plots + [temp_line] + [line for lines in bond_lines.values() for line in lines]

        frames = int(self.sim.total_time / self.sim.dt)
        ani = animation.FuncAnimation(fig, update_frame, frames=frames,
                                      interval=self.sim.dt * 1000, blit=False)
        plt.show()

    def plot_potential_energy(self):
        sim_data, analytical_data = self.sim.compute_potential_energy_data()
        if not sim_data:
            print("No potential energy data available.")
            return
        distances, potential_energies = zip(*sim_data)
        analytical_distances, analytical_potentials = zip(*analytical_data)
        plt.figure(figsize=(8, 6))
        plt.plot(analytical_distances, analytical_potentials, 'r-', label="Analytical LJ")
        plt.scatter(distances, potential_energies, c="blue", alpha=0.6, label="Simulated LJ")
        plt.xlabel("Distance (Å)")
        plt.ylabel("Potential Energy (eV)")
        plt.title("Potential Energy Plot")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_paths(self):
        plt.figure(figsize=(8, 6))
        for idx, path in self.sim.paths.items():
            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], label=f"Particle {idx}")
        plt.xlabel("X (Å)")
        plt.ylabel("Y (Å)")
        plt.title("Trajectories")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_all(self):
        self.plot_potential_energy()
        self.plot_paths()
