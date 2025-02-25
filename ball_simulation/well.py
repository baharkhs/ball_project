import numpy as np

class Well:
    def __init__(self, radius=2.0, height=4.0, wall_decay_length=0.05):
        """
        Initializes the Well object representing a cylindrical space for particle movement.

        Args:
            radius (float): Radius of the well in angstroms (1e-10 meters).
            height (float): Height of the well in angstroms.
            wall_decay_length (float): Decay length for wall repulsion (default 0.05 Å).
        """
        self.radius = radius  # Radius of the cylindrical boundary
        self.height = height  # Height of the well (z-axis boundary)
        self.wall_decay_length = wall_decay_length  # Decay length for wall repulsion

    def apply_pbc(self, ball):
        """
        Applies periodic boundary conditions (PBC) only in the z-direction.
        """
        if ball.position[2] > self.height:
            ball.position[2] -= self.height  # Wrap around to bottom
        elif ball.position[2] < 0:
            ball.position[2] += self.height  # Wrap around to top

    def apply_pbc_com(self, position, height, radius):
        """
        Applies PBC in z-direction and enforces a hard cylindrical boundary in x-y.
        """
        if position[2] > height:
            position[2] -= height
        elif position[2] < 0:
            position[2] += height
        r_xy = np.linalg.norm(position[:2])
        # Account for atom radius (0.1 Å) to ensure the entire atom stays within the wall
        effective_radius = radius - 0.1  # Subtract ball radius to check outer edge
        if r_xy > effective_radius:
            theta = np.arctan2(position[1], position[0])
            position[0] = effective_radius * np.cos(theta)
            position[1] = effective_radius * np.sin(theta)

    def compute_wall_repulsion_force(self, ball, repulsion_constant=500.0):
        """
        Computes a strong repulsive force near the cylindrical wall for individual atoms,
        repelling as soon as the outer edge of the atom touches the wall.
        """
        force = np.zeros(3)
        r_xy = np.linalg.norm(ball.position[:2])
        # Account for atom radius (0.1 Å) to repel when the outer edge touches the wall
        effective_radius = self.radius - 0.1  # Subtract ball radius for outer edge
        # Only apply repulsion if r_xy exceeds the threshold (effective_radius - wall_decay_length)
        if r_xy > effective_radius - self.wall_decay_length:
            overlap = r_xy - (effective_radius - self.wall_decay_length)
            if overlap > 0:
                theta = np.arctan2(ball.position[1], ball.position[0])
                normal_direction = np.array([np.cos(theta), np.sin(theta), 0.0])
                # Repel inward (negative force toward the center)
                force_magnitude = -repulsion_constant * np.exp(-overlap / self.wall_decay_length)  # Negative force
                force = force_magnitude * normal_direction  # Assign to the full 3D force vector
        return force

    def compute_wall_repulsion_force_com(self, position, repulsion_constant=500.0):
        """
        Computes strong wall repulsion force based on COM position, repelling as soon as the outer edge
        of any atom in the molecule touches the wall.
        """
        force = np.zeros(3)
        r_xy = np.linalg.norm(position[:2])
        # Account for atom radius (0.1 Å) to repel when the outer edge touches the wall
        effective_radius = self.radius - 0.1  # Subtract ball radius for outer edge
        # Only apply repulsion if r_xy exceeds the threshold (effective_radius - wall_decay_length)
        if r_xy > effective_radius - self.wall_decay_length:
            overlap = r_xy - (effective_radius - self.wall_decay_length)
            if overlap > 0:
                theta = np.arctan2(position[1], position[0])
                normal_direction = np.array([np.cos(theta), np.sin(theta), 0.0])
                # Repel inward (negative force toward the center)
                force_magnitude = -repulsion_constant * np.exp(-overlap / self.wall_decay_length)  # Negative force
                force = force_magnitude * normal_direction  # Assign to the full 3D force vector
        return force

    def plot_boundary(self, ax):
        """
        Draws the cylindrical boundary of the well for visualization.
        """
        theta = np.linspace(0, 2 * np.pi, 100)
        z = np.linspace(0, self.height, 50)
        theta, z = np.meshgrid(theta, z)
        x = self.radius * np.cos(theta)
        y = self.radius * np.sin(theta)
        ax.plot_wireframe(x, y, z, color="gray", alpha=0.3)

if __name__ == '__main__':
    # Example usage of Well class
    well = Well(radius=2.0, height=2.0, wall_decay_length=0.05)
    print(f"Well initialized with radius: {well.radius}, height: {well.height}, "
          f"wall_decay_length: {well.wall_decay_length}")
    # Test PBC
    from ball_simulation.ball import Ball
    ball = Ball(initial_position=[0.0, 0.0, 2.1])  # Above height
    well.apply_pbc(ball)
    print(f"After PBC (z=2.1): Position = {ball.position}")
    # Test wall repulsion
    ball = Ball(initial_position=[1.9, 0.0, 0.0])  # Near wall
    force = well.compute_wall_repulsion_force(ball)
    print(f"Wall repulsion force at [1.9, 0.0, 0.0]: {force}")