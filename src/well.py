# well.py
import numpy as np

class Well:
    """ Cylindrical simulation container. Calculates wall repulsion force. """
    def __init__(self, radius, height, wall_decay_length=0.05, repulsion_constant=500.0, atom_radius=0.1):
        self.radius = radius # nm
        self.height = height # nm
        self.wall_decay_length = wall_decay_length # nm
        # Assumes repulsion_constant scales force directly in kJ/mol/nm units
        self.repulsion_constant = repulsion_constant
        self.atom_radius = atom_radius # nm

    def compute_wall_repulsion_force(self, ball):
        """
        Computes wall repulsive force in kJ/mol/nm.
        """
        force_kj_mol_nm = np.zeros(3) # Units: kJ/mol/nm
        pos_x, pos_y = ball.position[0], ball.position[1]
        r_xy_sq = pos_x**2 + pos_y**2
        r_xy = np.sqrt(r_xy_sq) # nm

        effective_radius = self.radius - self.atom_radius # nm
        d_from_effective = r_xy - effective_radius # nm

        # Apply force if near or past the effective radius
        if d_from_effective > -self.wall_decay_length:
            force_mag_kj_mol_nm = 0.0
            outward_normal = np.array([0.0, 0.0, 0.0])
            # Calculate normal and magnitude only if not exactly on axis
            if r_xy != 0:
                 nx = pos_x / r_xy; ny = pos_y / r_xy
                 outward_normal[:2] = [nx, ny]
                 # Force magnitude using exponential decay, scaled by repulsion_constant
                 force_mag_kj_mol_nm = self.repulsion_constant * np.exp(d_from_effective / self.wall_decay_length)

            # Force vector points inward
            force_kj_mol_nm[:2] = -force_mag_kj_mol_nm * outward_normal[:2]

        return force_kj_mol_nm # Return force in kJ/mol/nm

    def plot_boundary(self, ax):
        """ Plots the cylindrical boundary wireframe. """
        theta_vals = np.linspace(0, 2 * np.pi, 100); z_vals = np.linspace(0, self.height, 50)
        theta_grid, z_grid = np.meshgrid(theta_vals, z_vals)
        x_grid = self.radius * np.cos(theta_grid); y_grid = self.radius * np.sin(theta_grid)
        ax.plot_wireframe(x_grid, y_grid, z_grid, color="gray", alpha=0.3, linewidth=0.5)