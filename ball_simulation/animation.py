import numpy as np


class Well:
    def __init__(self, radius=0.5, height=1.0):
        self.radius = radius
        self.height = height

    def apply_pbc(self, position):
        """Applies periodic boundary conditions in the z-direction only."""
        wrapped = False
        if position[2] > self.height:
            position[2] -= self.height  # Wrap to the bottom
            wrapped = True
        elif position[2] < 0:
            position[2] += self.height  # Wrap to the top
            wrapped = True
        return position, wrapped

    def apply_bounce(self, ball):
        """
        Applies a rigid bounce if the ball hits the x-y boundary,
        fully reversing the perpendicular velocity component upon contact.
        """
        # Check if ball is at or beyond the x-y boundary
        distance_from_center = np.linalg.norm(ball.position[:2])
        if distance_from_center >= self.radius:
            # Calculate the normal direction (outward from center of well in x-y plane)
            normal_direction = ball.position[:2] / distance_from_center
            # Reflect the velocity component perpendicular to the wall
            ball.velocity[:2] -= 2 * np.dot(ball.velocity[:2], normal_direction) * normal_direction
            # Move the ball back just within the boundary to prevent it from sticking
            ball.position[:2] = normal_direction * (self.radius - 1e-6)

        return ball.velocity


class Ball:
    def __init__(self, mass=1.0, initial_position=None, initial_velocity=None):
        self.mass = mass
        self.position = np.array(initial_position) if initial_position is not None else np.array([0, 0, 0])
        self.velocity = np.array(initial_velocity if initial_velocity else [0, 0, 0])
        self.initial_velocity = self.velocity.copy()
        self.force = np.zeros(3)
        #self.gravity = gravity
        self.path_x, self.path_y, self.path_z = [], [], []
        self.skip_path_update = False

    def calculate_kinetic_energy(self):
        """Calculates the kinetic energy of the ball."""
        return 0.5 * self.mass * np.dot(self.velocity, self.velocity)

    def apply_forces(self):
        """Applies gravitational force to the ball."""
        #self.force = self.mass * self.gravity
        pass

    def update_velocity_position(self, dt):
        """Updates the velocity and position, with kinetic energy diagnostic check."""
        #acceleration = self.force / self.mass
        #self.velocity += acceleration * dt
        self.position += self.velocity * dt

        # Diagnostic check
        #kinetic_energy = self.calculate_kinetic_energy()
        #print(f"Kinetic Energy after update: {kinetic_energy:.4f}")
        #print(f"Velocity after update: {self.velocity}")

    def update_path(self):
        """Records the ball's position for plotting, skipping updates during PBC transitions."""
        if self.skip_path_update:
            # Clear the path if a PBC transition occurred to avoid vertical lines
            self.path_x.clear()
            self.path_y.clear()
            self.path_z.clear()
        else:
            # Append the current position if no PBC transition happened
            self.path_x.append(self.position[0])
            self.path_y.append(self.position[1])
            self.path_z.append(self.position[2])

        # Reset the skip flag for the next update
        self.skip_path_update = False

class Simulation:
    def __init__(self, well_radius=0.5, well_height=1.0, total_time=10.0, dt=0.01):
        self.well = Well(well_radius, well_height)
        self.dt = dt
        self.total_time = total_time
        self.balls = []

    def add_ball(self, mass=1.0, initial_position=None, initial_velocity=None):
        """Adds a ball to the simulation."""
        ball = Ball(mass=mass, initial_position=initial_position, initial_velocity=initial_velocity)
        self.balls.append(ball)

    def update(self):
        for ball in self.balls:
            # Apply forces and update velocity and position
            ball.apply_forces()
            ball.update_velocity_position(self.dt)

            # Apply bounce in x-y direction
            ball.velocity = self.well.apply_bounce(ball)

            # Apply PBC in z-direction
            ball.position, wrapped = self.well.apply_pbc(ball.position)

            # Set flag to skip path update if wrapped in the z-direction
            ball.skip_path_update = wrapped
            ball.update_path()
