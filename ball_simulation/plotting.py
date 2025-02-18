import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
class SimulationPlotter:
    def __init__(self, simulation):
        self.sim = simulation

    def animate_simulation(self):
        """Sets up and runs the 3D animation for the simulation."""
        fig = plt.figure(figsize=(8, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
        ax_sim = fig.add_subplot(gs[0], projection='3d')
        ax_temp = fig.add_subplot(gs[1])

        # Configure simulation axes.
        ax_sim.set_xlim(-self.sim.well.radius, self.sim.well.radius)
        ax_sim.set_ylim(-self.sim.well.radius, self.sim.well.radius)
        ax_sim.set_zlim(0, self.sim.well.height)
        ax_sim.set_title("Simulation")
        ax_sim.set_xlabel("X (Å)")
        ax_sim.set_ylabel("Y (Å)")
        ax_sim.set_zlabel("Z (Å)")
        self.sim.well.plot_boundary(ax_sim)

        # Configure temperature plot.
        ax_temp.set_title("System Temperature Over Time")
        ax_temp.set_xlabel("Step")
        ax_temp.set_ylabel("Temperature (K)")
        temp_line, = ax_temp.plot([], [], color="blue")
        ax_temp.grid(True)

        # Create scatter plots for each ball.
        ball_plots = [ax_sim.plot([], [], [], 'o', color=b.color, markersize=b.size)[0] for b in self.sim.balls]

        def update_frame(frame):
            self.sim.update()
            for i, ball in enumerate(self.sim.balls):
                ball_plots[i].set_data([ball.position[0]], [ball.position[1]])
                ball_plots[i].set_3d_properties([ball.position[2]])
            steps = np.arange(len(self.sim.temperature_history))
            temp_line.set_data(steps, self.sim.temperature_history)
            ax_temp.set_xlim(0, max(1, len(steps)))
            if self.sim.temperature_history:
                ax_temp.set_ylim(0, max(self.sim.temperature_history) + 50)
            return ball_plots + [temp_line]

        frames = int(self.sim.total_time / self.sim.dt)
        ani = animation.FuncAnimation(fig, update_frame, frames=frames, interval=self.sim.dt * 1000, blit=False)
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