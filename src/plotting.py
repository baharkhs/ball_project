import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


class SimulationPlotter:
    def __init__(self, simulation):
        """
        Initializes the plotter with a Simulation object.
        Sets up a Matplotlib figure with two subplots:
         - A 3D axis (ax_sim) for displaying particle positions and bonds.
         - A 2D axis (ax_temp) for plotting system temperature over time.
        """
        self.sim = simulation
        self.fig = plt.figure(figsize=(10, 5))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[2, 1])
        self.ax_sim = self.fig.add_subplot(gs[0], projection='3d')
        self.ax_temp = self.fig.add_subplot(gs[1])
        self._init_draw()

    def _init_draw(self):
        """Initializes the drawing axes for simulation and temperature."""
        # Set simulation 3D axis limits based on well dimensions.
        self.ax_sim.set_xlim(-self.sim.well.radius, self.sim.well.radius)
        self.ax_sim.set_ylim(-self.sim.well.radius, self.sim.well.radius)
        self.ax_sim.set_zlim(0, self.sim.well.height)
        self.ax_sim.set_xlabel("X (Å)")
        self.ax_sim.set_ylabel("Y (Å)")
        self.ax_sim.set_zlabel("Z (Å)")
        self.ax_sim.set_title("Water Molecule Simulation")

        # Plot the well boundary if desired (assuming your Well class provides a plot_boundary method).
        if hasattr(self.sim.well, "plot_boundary"):
            self.sim.well.plot_boundary(self.ax_sim)

        # Configure the temperature subplot.
        self.ax_temp.set_title("System Temperature Over Time")
        self.ax_temp.set_xlabel("Step")
        self.ax_temp.set_ylabel("Temperature (K)")
        self.ax_temp.grid(True)
        self.temp_line, = self.ax_temp.plot([], [], lw=2, color="blue")

    def _draw_frame(self, frame):
        """
        Updates the plot for a single frame:
         - Advances the simulation one time step.
         - Updates the 3D scatter for particle positions.
         - Updates the 2D temperature plot.
        """
        # Advance simulation by one update step.
        self.sim.update()

        # Clear the 3D axis and redraw the well boundary.
        self.ax_sim.cla()
        self.ax_sim.set_xlim(-self.sim.well.radius, self.sim.well.radius)
        self.ax_sim.set_ylim(-self.sim.well.radius, self.sim.well.radius)
        self.ax_sim.set_zlim(0, self.sim.well.height)
        self.ax_sim.set_xlabel("X (Å)")
        self.ax_sim.set_ylabel("Y (Å)")
        self.ax_sim.set_zlabel("Z (Å)")
        self.ax_sim.set_title("Water Molecule Simulation")
        if hasattr(self.sim.well, "plot_boundary"):
            self.sim.well.plot_boundary(self.ax_sim)

        # Plot each Ball as a point in 3D.
        positions = np.array([ball.position for ball in self.sim.balls])
        self.ax_sim.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=60, c=[
            ball.color for ball in self.sim.balls
        ])

        # Optionally, draw bonds between atoms belonging to the same molecule.
        for mol_id, indices in self.sim.molecules.items():
            # For water, draw bonds from oxygen to each hydrogen.
            o_idx = indices["O"]
            h1_idx = indices["H1"]
            h2_idx = indices["H2"]
            O_pos = self.sim.balls[o_idx].position
            H1_pos = self.sim.balls[h1_idx].position
            H2_pos = self.sim.balls[h2_idx].position
            self.ax_sim.plot([O_pos[0], H1_pos[0]], [O_pos[1], H1_pos[1]], [O_pos[2], H1_pos[2]], c="green", lw=2)
            self.ax_sim.plot([O_pos[0], H2_pos[0]], [O_pos[1], H2_pos[1]], [O_pos[2], H2_pos[2]], c="green", lw=2)

        # Update temperature plot.
        steps = np.arange(len(self.sim.temperature_history))
        self.temp_line.set_data(steps, self.sim.temperature_history)
        self.ax_temp.set_xlim(0, max(10, len(steps)))
        if self.sim.temperature_history:
            self.ax_temp.set_ylim(0, max(self.sim.temperature_history) + 50)
        return self.temp_line,

    def animate_simulation(self):
        """
        Starts the animation of the simulation using FuncAnimation.
        This continuously updates both the 3D simulation view and the 2D temperature plot.
        """
        total_frames = int(self.sim.total_time / self.sim.dt)
        ani = FuncAnimation(self.fig, self._draw_frame, frames=total_frames, interval=50, blit=False)
        plt.show()
