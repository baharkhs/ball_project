import numpy as np

class Well:
    def __init__(self, radius=2.0, height=4.0):
        self.radius = radius
        self.height = height

    def apply_pbc(self, ball):
        """
        Applies periodic boundary conditions (PBC) only in the z-direction.
        """
        if ball.position[2] > self.height:
            ball.position[2] -= self.height
        elif ball.position[2] < 0:
            ball.position[2] += self.height

    def apply_pbc_com(self, position, height, radius):
        """
        Applies PBC in z-direction and enforces a hard cylindrical boundary in x-y.
        """
        if position[2] > height:
            position[2] -= height
        elif position[2] < 0:
            position[2] += height
        r_xy = np.linalg.norm(position[:2])
        if r_xy > radius:
            theta = np.arctan2(position[1], position[0])
            position[0] = radius * np.cos(theta)
            position[1] = radius * np.sin(theta)

    def compute_wall_repulsion_force(self, ball, repulsion_constant=500.0, wall_decay_length=0.05):
        """
        Computes a strong repulsive force near the cylindrical wall for individual atoms.
        """
        force = np.zeros(3)
        r_xy = np.linalg.norm(ball.position[:2])
        if r_xy > self.radius - wall_decay_length:
            overlap = r_xy - (self.radius - wall_decay_length)
            if overlap > 0:
                normal_direction = ball.position[:2] / r_xy
                force_magnitude = repulsion_constant * np.exp(-overlap / wall_decay_length)
                force[:2] = -force_magnitude * normal_direction
        return force

    def compute_wall_repulsion_force_com(self, position, repulsion_constant=500.0, wall_decay_length=0.05):
        """
        Computes strong wall repulsion force based on COM position (for consistency).
        """
        force = np.zeros(3)
        r_xy = np.linalg.norm(position[:2])
        if r_xy > self.radius - wall_decay_length:
            overlap = r_xy - (self.radius - wall_decay_length)
            if overlap > 0:
                normal_direction = position[:2] / r_xy
                force_magnitude = repulsion_constant * np.exp(-overlap / wall_decay_length)
                force[:2] = -force_magnitude * normal_direction
        return force

    def plot_boundary(self, ax):
        theta = np.linspace(0, 2 * np.pi, 100)
        z = np.linspace(0, self.height, 50)
        theta, z = np.meshgrid(theta, z)
        x = self.radius * np.cos(theta)
        y = self.radius * np.sin(theta)
        ax.plot_wireframe(x, y, z, color="gray", alpha=0.3)