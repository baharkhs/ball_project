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
        """Applies a rigid bounce if the ball hits the x-y boundary, reversing perpendicular velocity."""
        distance_from_center = np.linalg.norm(ball.position[:2])
        if distance_from_center >= self.radius:
            normal_direction = ball.position[:2] / distance_from_center
            ball.velocity[:2] -= 2 * np.dot(ball.velocity[:2], normal_direction) * normal_direction
            ball.position[:2] = normal_direction * (self.radius - 1e-6)  # Keep within bounds

        return ball.velocity


class Ball:
    def __init__(self, mass=1.0, initial_position=None, initial_velocity=None):
        self.mass = mass
        self.position = np.array(initial_position) if initial_position is not None else np.array([0, 0, 0])
        self.velocity = np.array(initial_velocity if initial_velocity else [0, 0, 0])
        self.initial_velocity = self.velocity.copy()
        self.path_segments = []  # Stores path segments as lists of [x, y, z] positions
        self.current_path_segment = {"x": [], "y": [], "z": []}
        self.force = np.zeros(3)
        #self.gravity = gravity  # Placeholder for future gravity implementation
        self.skip_path_update = False

    def calculate_kinetic_energy(self):
        """Calculates the kinetic energy of the ball."""
        return 0.5 * self.mass * np.dot(self.velocity, self.velocity)

    def apply_forces(self):
        """Placeholder for applying forces to the ball (e.g., gravity)."""
        #self.force = self.mass * self.gravity  # Placeholder for future gravity application
        pass

    def update_velocity_position(self, dt):
        """Updates the velocity and position of the ball."""
        # acceleration = self.force / self.mass
        # self.velocity += acceleration * dt  # Placeholder for velocity update if forces are used
        self.position += self.velocity * dt

    def update_path(self):
        """Records the ball's position for plotting, and segments path on PBC transitions."""
        if self.skip_path_update:
            if self.current_path_segment["x"]:
                self.path_segments.append(self.current_path_segment)
            self.current_path_segment = {"x": [], "y": [], "z": []}
            self.skip_path_update = False
        else:
            self.current_path_segment["x"].append(self.position[0])
            self.current_path_segment["y"].append(self.position[1])
            self.current_path_segment["z"].append(self.position[2])

    def finalize_path(self):
        """Finalizes and stores the current path segment at the end of the simulation."""
        if self.current_path_segment["x"]:
            self.path_segments.append(self.current_path_segment)

    def get_path_segments(self):
        """Returns all path segments recorded."""
        return self.path_segments + [self.current_path_segment]


class Simulation:
    def __init__(self, well_radius=0.5, well_height=1.0, total_time=10.0, dt=0.01, movement_type="newtonian"):
        self.well = Well(well_radius, well_height)
        self.dt = dt
        self.total_time = total_time
        self.balls = []
        self.movement_type = movement_type  # Define system-wide movement type for all balls

    def set_movement_type(self, movement_type="newtonian"):
        """Sets the movement type for all balls in the simulation."""
        self.movement_type = movement_type

    def add_ball(self, mass=1.0, initial_position=None, initial_velocity=None):
        """Adds a ball to the simulation with the specified properties."""
        ball = Ball(mass=mass, initial_position=initial_position, initial_velocity=initial_velocity)
        self.balls.append(ball)

    def apply_movement(self, ball):
        """Applies movement to a ball based on the set movement type."""
        if self.movement_type == "newtonian":
            ball.update_velocity_position(self.dt)
        elif self.movement_type == "monte_carlo":
            self.apply_monte_carlo_movement(ball)
        else:
            raise ValueError(f"Unknown movement type: {self.movement_type}")

    def apply_monte_carlo_movement(self, ball):
        """Applies Monte Carlo random walk movement to the ball."""
        random_step = np.random.uniform(-0.05, 0.05, size=3)
        ball.position += random_step * self.dt

    def update(self):
        """Updates each ball's position, velocity, and handles boundary conditions."""
        for ball in self.balls:
            ball.apply_forces()
            self.apply_movement(ball)  # Apply movement based on system movement type

            ball.velocity = self.well.apply_bounce(ball)
            ball.position, wrapped = self.well.apply_pbc(ball.position)

            ball.skip_path_update = wrapped
            ball.update_path()

    def finalize_simulation(self):
        """Finalizes the path for each ball at the end of the simulation."""
        for ball in self.balls:
            ball.finalize_path()
