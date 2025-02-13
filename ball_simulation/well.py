import numpy as np
class Well:
    def __init__(self, radius=5.0, height=10.0):
        """
        Initializes the Well object representing a cylindrical space for particle movement.

        Args:
            radius (float): Radius of the well in angstroms (1e-10 meters).
            height (float): Height of the well in angstroms.
        """
        self.radius = radius  # Radius of the cylindrical boundary
        self.height = height  # Height of the well (z-axis boundary)

    def apply_pbc(self, ball):
        """
        Applies periodic boundary conditions (PBC) for the particle at the cylindrical x-y boundary and z-boundaries.

        Args:
            ball (Ball): The Ball object whose position will be adjusted to apply PBC.

        Returns:
            None
        """

        # Apply PBC in the z-direction (top and bottom boundaries)
        if ball.position[2] > self.height:
            ball.position[2] -= self.height  # Wrap around to bottom
        elif ball.position[2] < 0:
            ball.position[2] += self.height  # Wrap around to top

        # Apply PBC in the cylindrical x-y plane
        distance_from_center = np.linalg.norm(ball.position[:2])
        if distance_from_center > self.radius:  # Project back into the cylindrical boundary
            theta = np.arctan2(ball.position[1], ball.position[0])  # Compute angle
            ball.position[0] = self.radius * np.cos(theta)
            ball.position[1] = self.radius * np.sin(theta)

    def compute_wall_repulsion_force(self, ball, repulsion_constant=50.0, wall_decay_length=0.2):
        """
        Computes a soft repulsive force near the cylindrical and z-boundaries of the well.

        Args:
            ball (Ball): The Ball object to compute forces for.
            repulsion_constant (float): Strength of the repulsion force.
            wall_decay_length (float): Decay length controlling force range.

        Returns:
            np.array: The repulsion force vector acting on the ball.
        """
        force = np.zeros(3)

        # Repulsion from cylindrical x-y boundary
        distance_from_center = np.linalg.norm(ball.position[:2])
        if distance_from_center > self.radius - wall_decay_length:  # Near boundary
            overlap = distance_from_center - (self.radius - wall_decay_length)
            if overlap > 0:
                normal_direction = ball.position[:2] / distance_from_center
                force[:2] = -repulsion_constant * np.exp(-overlap / wall_decay_length) * normal_direction

        # Repulsion from bottom z-boundary
        #if ball.position[2] < wall_decay_length:  # Near bottom wall
         #   overlap = wall_decay_length - ball.position[2]
          #  force[2] += repulsion_constant * np.exp(-overlap / wall_decay_length)

        # Repulsion from top z-boundary
        #if ball.position[2] > self.height - wall_decay_length:  # Near top wall
         #   overlap = ball.position[2] - (self.height - wall_decay_length)
          #  force[2] -= repulsion_constant * np.exp(-overlap / wall_decay_length)

        return force

    def plot_boundary(self, ax):
        """
        Draws the cylindrical boundary of the well for visualization.

        Args:
            ax (Axes3D): Matplotlib 3D axis to plot the well boundary.
        """
        theta = np.linspace(0, 2 * np.pi, 100)
        z = np.linspace(0, self.height, 50)
        theta, z = np.meshgrid(theta, z)
        x = self.radius * np.cos(theta)
        y = self.radius * np.sin(theta)
        ax.plot_wireframe(x, y, z, color="gray", alpha=0.3)