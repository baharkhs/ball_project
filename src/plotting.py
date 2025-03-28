# plotting.py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class SimulationPlotter:
    """
    Handles the visualization of the molecular dynamics simulation.

    Creates a Matplotlib figure with a 3D view of the particles and container,
    and a 2D plot of the system's temperature over time. Animates the simulation
    by repeatedly updating these plots. Assumes simulation uses nm and fs internally.
    """
    def __init__(self, simulation, plot_config=None):
        """
        Initializes the plotter with a Simulation object and optional plotting configuration.

        Args:
            simulation (Simulation): The simulation instance to visualize.
            plot_config (dict, optional): Dictionary with plotting parameters (e.g., colors, sizes).
        """
        self.sim = simulation
        # Use plotting config from simulation config file or use defaults
        self.plot_cfg = plot_config if plot_config is not None else {
            "figure_size": [10, 5], "width_ratios": [2, 1], "bond_color": "gray",
            "bond_linewidth": 1, "boundary_color": "gray", "boundary_alpha": 0.3,
            "temperature_color": "blue", "temperature_linewidth": 2,
            "temperature_buffer": 50
        }

        self.fig = plt.figure(figsize=self.plot_cfg.get("figure_size", [10, 5]))
        gs = self.fig.add_gridspec(1, 2, width_ratios=self.plot_cfg.get("width_ratios", [2, 1]))
        self.ax_sim = self.fig.add_subplot(gs[0], projection='3d')
        self.ax_temp = self.fig.add_subplot(gs[1])

        # Internal state for updatable plot elements
        self.scatter_particles = None # The scatter plot object for atoms
        self.bond_lines = []          # List holding bond line objects
        self.path_lines = {}          # Dict holding path line objects {ball_index: [lines]}
        self.temp_line = None         # The temperature line object

        self._init_plot_layout() # Setup initial plot appearance

    def _init_plot_layout(self):
        """ Sets up the initial appearance and labels of the plot axes. """
        # Configure the 3D simulation axis
        radius_nm = self.sim.well.radius
        height_nm = self.sim.well.height
        self.ax_sim.set_xlim(-radius_nm, radius_nm)
        self.ax_sim.set_ylim(-radius_nm, radius_nm)
        self.ax_sim.set_zlim(0, height_nm)
        self.ax_sim.set_xlabel("X (nm)") # Units: nm
        self.ax_sim.set_ylabel("Y (nm)") # Units: nm
        self.ax_sim.set_zlabel("Z (nm)") # Units: nm
        self.ax_sim.set_title("Molecular Simulation")

        # Plot the static well boundary using the Well's method
        if hasattr(self.sim.well, "plot_boundary"):
             self.sim.well.plot_boundary(self.ax_sim)

        # Configure the 2D temperature axis
        self.ax_temp.set_title("System Temperature")
        self.ax_temp.set_xlabel("Time Step") # X-axis represents simulation steps
        self.ax_temp.set_ylabel("Temperature (K)") # Y-axis is temperature in Kelvin
        self.ax_temp.grid(True)
        # Initialize temperature line object with empty data
        self.temp_line, = self.ax_temp.plot([], [],
                                             lw=self.plot_cfg.get("temperature_linewidth", 2),
                                             color=self.plot_cfg.get("temperature_color", "blue"))

    def _draw_frame(self, frame_num):
        """
        Updates the plot elements for a single animation frame by running one
        simulation step and redrawing dynamic elements.

        Args:
            frame_num (int): The current animation frame number.

        Returns:
            list: A list of updated matplotlib artists.
        """
        # 1. Advance the simulation by one step (using MD update)
        self.sim.update()

        # 2. Update particle positions (scatter plot)
        positions = np.array([ball.position for ball in self.sim.balls]) # Get all positions (N, 3)
        colors = [ball.color for ball in self.sim.balls]
        sizes = [ball.size * 5 for ball in self.sim.balls] # Scale default size

        # Create scatter plot on first frame, update positions on subsequent frames
        if self.scatter_particles is None:
            self.scatter_particles = self.ax_sim.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                                                         s=sizes, c=colors, depthshade=True)
        else:
            # Update positions efficiently
            self.scatter_particles._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])

        # 3. Update bond lines (remove old, draw new)
        for line in self.bond_lines: line.remove() # Clear previous bonds
        self.bond_lines.clear()
        bond_color = self.plot_cfg.get("bond_color", "gray")
        bond_lw = self.plot_cfg.get("bond_linewidth", 1)
        # Iterate through registered molecules to draw bonds
        for mol_id, indices in self.sim.molecules.items():
            # Specific logic for water molecules (O-H1, O-H2)
            if "O" in indices and "H1" in indices and "H2" in indices:
                o_idx, h1_idx, h2_idx = indices["O"], indices["H1"], indices["H2"]
                O_pos = self.sim.balls[o_idx].position
                H1_pos = self.sim.balls[h1_idx].position
                H2_pos = self.sim.balls[h2_idx].position
                # Draw O-H1 bond
                line1, = self.ax_sim.plot([O_pos[0], H1_pos[0]], [O_pos[1], H1_pos[1]], [O_pos[2], H1_pos[2]],
                                          color=bond_color, lw=bond_lw)
                # Draw O-H2 bond
                line2, = self.ax_sim.plot([O_pos[0], H2_pos[0]], [O_pos[1], H2_pos[1]], [O_pos[2], H2_pos[2]],
                                          color=bond_color, lw=bond_lw)
                self.bond_lines.extend([line1, line2]) # Keep track of drawn lines

        # 4. Update particle paths (Optional - can impact performance significantly)
        # Keep commented out unless path visualization is essential and performance allows.
        '''
        path_color = self.plot_cfg.get("path_color", "blue") # Example config
        path_lw = self.plot_cfg.get("path_linewidth", 0.5)
        path_alpha = self.plot_cfg.get("path_alpha", 0.5)
        for i, ball in enumerate(self.sim.balls):
            if i in self.path_lines: # Remove old path segments for this ball
                for line in self.path_lines[i]: line.remove()
            self.path_lines[i] = [] # Reset list for this ball
            segments = ball.get_path_segments() # Get current trajectory segments
            for segment in segments:
                if len(segment["x"]) > 1: # Check if segment has enough points
                    line, = self.ax_sim.plot(segment["x"], segment["y"], segment["z"],
                                             color=path_color, lw=path_lw, alpha=path_alpha)
                    self.path_lines[i].append(line) # Store the line object
        '''

        # 5. Update temperature plot
        steps = np.arange(self.sim.current_step) # X-axis data: time steps
        temps = self.sim.temperature_history    # Y-axis data: recorded temperatures (K)
        # Ensure data arrays have matching lengths before plotting
        if len(steps) == len(temps):
            self.temp_line.set_data(steps, temps) # Update the line data
            # Dynamically adjust plot limits for visibility
            self.ax_temp.set_xlim(0, max(10, self.sim.current_step)) # Maintain minimum width
            if temps: # Check if temperature history is not empty
                temp_max = max(temps)
                temp_buffer = self.plot_cfg.get("temperature_buffer", 50)
                self.ax_temp.set_ylim(0, temp_max + temp_buffer) # Set Y limits with buffer
            # Handle case where history might be empty initially
            elif self.sim.target_temperature > 0:
                 self.ax_temp.set_ylim(0, self.sim.target_temperature * 1.5)


        # Return list of artists that were modified in this frame
        artists = [self.scatter_particles, self.temp_line] + self.bond_lines
        # If path plotting is enabled, add path lines to the returned list:
        # for ball_idx in self.path_lines: artists.extend(self.path_lines[ball_idx])
        return artists

    def animate_simulation(self, interval_ms=50, save_path=None):
        """
        Starts the simulation animation using Matplotlib's FuncAnimation.

        Args:
            interval_ms (int): Delay between frames in milliseconds. Controls animation speed.
            save_path (str, optional): File path to save the animation (e.g., 'sim.mp4').
                                       Requires ffmpeg. If None, shows interactively.
        """
        # Calculate total frames based on simulation time (fs) and timestep (fs)
        if self.sim.dt_fs <= 0:
             total_frames = 0
             print("Warning: Simulation dt is zero or negative, cannot animate.")
        else:
             total_frames = int(self.sim.total_time_fs / self.sim.dt_fs)

        print(f"Starting animation for {total_frames} frames...")

        # Create the animation object. blit=False is often more reliable for complex plots.
        ani = FuncAnimation(self.fig, self._draw_frame, frames=total_frames,
                            interval=interval_ms, blit=False, repeat=False)

        # Handle saving or interactive display
        if save_path:
            print(f"Attempting to save animation to {save_path}...")
            # Use a try block here specifically for external library calls like saving
            try:
                ani.save(save_path, writer='ffmpeg', dpi=150) # Example writer and DPI
                print(f"Animation saved successfully to {save_path}.")
            except Exception as e:
                # Catch potential errors during saving (e.g., ffmpeg not found)
                print(f"Error saving animation: {e}")
                print("Displaying interactively instead.")
                plt.tight_layout()
                plt.show() # Fallback to interactive display
        else:
            plt.tight_layout() # Adjust layout before showing
            plt.show() # Show interactive plot window

        # Finalize paths after animation is complete or window is closed
        for ball in self.sim.balls:
             ball.finalize_path()
        print("Simulation and plotting finished.")