import numpy as np

class Well:
    def __init__(self, radius, height, wall_decay_length=0.05, repulsion_constant=500.0, atom_radius=0.1):
        self.radius = radius
        self.height = height
        self.wall_decay_length = wall_decay_length
        self.repulsion_constant = repulsion_constant
        self.atom_radius = atom_radius

    def apply_pbc(self, ball):
        # Wrap the z-coordinate within the well height.
        ball.position[2] %= self.height
        # Constrain the x-y coordinates within the cylinder (if needed, scaling them down).
        r_xy = np.linalg.norm(ball.position[:2])
        if r_xy > self.radius:
            ball.position[:2] *= self.radius / r_xy

    def compute_wall_repulsion_force(self, ball):
        """
        Computes a repulsive force for a ball if it gets too close to the cylindrical wall.
        The force is calculated based on an exponential decay that depends on how far past a threshold the ball is.
        """
        force = np.zeros(3)
        # Compute radial distance in x-y plane.
        r_xy = np.linalg.norm(ball.position[:2])
        # Effective boundary: well radius adjusted for the ball's radius.
        effective_radius = self.radius - self.atom_radius
        # Check if the ball is within a zone near the boundary.
        if r_xy > effective_radius - self.wall_decay_length:
            # Determine the overlap past the safe zone.
            overlap = r_xy - (effective_radius - self.wall_decay_length)
            if overlap > 0:
                theta = np.arctan2(ball.position[1], ball.position[0])
                # The normal points inward.
                normal = np.array([np.cos(theta), np.sin(theta), 0.0])
                # Compute a repulsion magnitude that decays exponentially with overlap.
                force_magnitude = -self.repulsion_constant * np.exp(-overlap / self.wall_decay_length)
                force[:2] = force_magnitude * normal[:2]
        return force

    def plot_boundary(self, ax):
        """
        Plots the cylindrical boundary of the well for visualization.
        """
        theta = np.linspace(0, 2 * np.pi, 100)
        z = np.linspace(0, self.height, 50)
        theta, z = np.meshgrid(theta, z)
        x = self.radius * np.cos(theta)
        y = self.radius * np.sin(theta)
        ax.plot_wireframe(x, y, z, color="gray", alpha=0.3)
