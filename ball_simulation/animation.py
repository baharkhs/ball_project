import numpy as np

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

    def apply_pbc(self, position):
        """
        Applies periodic boundary conditions (PBC) along the z-axis only, wrapping particles
        between the top and bottom boundaries of the well.

        This method checks if the particle's z-coordinate exceeds the boundaries of the well
        (either above the top or below the bottom). If so, the z-coordinate is adjusted to wrap
        the particle around the opposite boundary. A flag is returned to indicate if the particle
        underwent a PBC transition.

        Args:
            position (np.array): The [x, y, z] position of the particle in angstroms.

        Returns:
            tuple:
                np.array: The modified position after applying PBC.
                bool: True if the particle underwent a PBC transition, False otherwise.
        """
        wrapped = False  # Flag to track if a PBC transition occurred

        # Check if the particle is above the top boundary
        if position[2] > self.height:
            position[2] -= self.height  # Wrap to the bottom
            wrapped = True

        # Check if the particle is below the bottom boundary
        elif position[2] < 0:
            position[2] += self.height  # Wrap to the top
            wrapped = True

        return position, wrapped

    def apply_bounce(self, ball):
        """
        Applies a rigid bounce if the particle reaches the x-y boundary, reversing its velocity component
        perpendicular to the boundary.

        Args:
            ball (Ball): The Ball object to check and adjust if it hits the boundary.

        Returns:
            np.array: Updated velocity of the ball after bounce, if applicable.
        """
        # Calculate the distance from the center to see if it's at or beyond the boundary
        distance_from_center = np.linalg.norm(ball.position[:2])
        if distance_from_center >= self.radius:
            # Determine the normal direction for reflection
            normal_direction = ball.position[:2] / distance_from_center
            # Reflect velocity across the boundary normal direction
            ball.velocity[:2] -= 2 * np.dot(ball.velocity[:2], normal_direction) * normal_direction
            # Adjust position to keep within the boundary (adjustment based on angstrom scale)
            ball.position[:2] = normal_direction * (self.radius - 1e-10)

        return ball.velocity



class Ball:
    def __init__(self, mass=18.0, initial_position=None, initial_velocity=None):
        """
        Represents a single particle with mass, position, velocity, and temperature attributes.

        Args:
            mass (float): Mass of the particle in atomic mass units (amu).
            initial_position (array-like): Initial [x, y, z] position in angstroms.
            initial_velocity (array-like): Initial [vx, vy, vz] velocity in angstroms per fs.
        """
        self.mass = mass  # Mass in atomic mass units (amu)
        self.position = np.array(initial_position) if initial_position is not None else np.array([0, 0, 0])
        self.velocity = np.array(initial_velocity if initial_velocity else [0, 0, 0])
        self.initial_velocity = self.velocity.copy()  # Store original velocity for calculations
        self.path_segments = []  # Stores path segments as lists of [x, y, z] positions
        self.current_path_segment = {"x": [], "y": [], "z": []}
        # self.gravity = gravity  # Placeholder for future gravity implementation
        self.skip_path_update = False  # For handling periodic boundary condition resets
        self.temperature = self.calculate_temperature()  # Temperature of the particle (K)
        self.force = np.zeros(3)  # Total force acting on the ball
        self.radius = 0.1  # Radius of the ball for collisions

    def compute_repulsion_force(self, other, repulsion_constant=1.0):
        """Compute the repulsive force exerted by another ball."""
        displacement = self.position - other.position
        distance = np.linalg.norm(displacement)
        if distance == 0 or distance > 2 * self.radius:  # Ignore if too far or overlapping
            return np.zeros(3)

        # Cap the repulsion force to avoid excessive acceleration
        max_force = 10.0
        force_magnitude = min(repulsion_constant / (distance ** 2), max_force)
        force_direction = displacement / distance  # Unit vector
        return force_magnitude * force_direction

    def calculate_kinetic_energy(self):
        """
        Calculates and returns the kinetic energy (KE) in atomic units.

        Returns:
            float: Kinetic energy in (amu * (angstrom/fs)^2).
        """
        return 0.5 * self.mass * np.dot(self.velocity, self.velocity)

    def calculate_temperature(self):
        """
        Calculates and updates the particle's temperature based on its kinetic energy.

        Returns:
            float: Temperature in Kelvin (K).
        """
        # Boltzmann constant (in units compatible with angstrom/fs and amu)
        k_B = 0.0083144621  # Boltzmann constant in atomic units
        kinetic_energy = self.calculate_kinetic_energy()
        self.temperature = (2 / 3) * (kinetic_energy / k_B)
        return self.temperature

    def apply_forces(self):
        """
        Placeholder for applying forces to the ball (e.g., gravity).
        """
        # self.force = self.mass * self.gravity  # Placeholder for future gravity application
        pass

    def update_velocity_position(self, dt):
        """
        Updates the particle's position based on velocity and time step.

        Args:
            dt (float): Time step in femtoseconds.
        """
        # acceleration = self.force / self.mass  # Uncomment if force-based acceleration is applied
        # self.velocity += acceleration * dt  # Update velocity if forces are active
        self.position += self.velocity * dt  # Position update based on current velocity

    def update_path(self):
        """Records the particle's position for path visualization, handling PBC transitions."""
        if self.skip_path_update:
            # Start a new segment if skip_path_update is set
            if self.current_path_segment["x"]:
                self.path_segments.append(self.current_path_segment)
            self.current_path_segment = {"x": [], "y": [], "z": []}
            self.skip_path_update = False  # Reset the flag
        else:
            # Append current position to the ongoing segment
            self.current_path_segment["x"].append(self.position[0])
            self.current_path_segment["y"].append(self.position[1])
            self.current_path_segment["z"].append(self.position[2])

    def finalize_path(self):
        """Finalizes and stores the current path segment at the end of the simulation."""
        if self.current_path_segment["x"]:
            self.path_segments.append(self.current_path_segment)

    def get_path_segments(self):
        """Returns all path segments for visualization."""
        return self.path_segments + [self.current_path_segment]

    def finalize_simulation(self):
        """
        Finalizes the path for each ball at the end of the simulation.
        """
        for ball in self.balls:
            ball.finalize_path()


class Simulation:
    def __init__(self, well_radius=0.5, well_height=1.0, total_time=10.0, dt=0.01, movement_type="newtonian"):
        """
        Initializes the simulation environment, parameters, and balls.

        Args:
            well_radius (float): Radius of the cylindrical well in angstroms.
            well_height (float): Height of the well in angstroms.
            total_time (float): Total simulation time in femtoseconds.
            dt (float): Time step in femtoseconds.
            movement_type (str): Movement type for particles; "newtonian" or "monte_carlo".
        """
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

    def calculate_average_velocity(self):
        """Calculates the average velocity magnitude of all balls in the simulation."""
        if self.balls:
            velocities = [np.linalg.norm(ball.velocity) for ball in self.balls]
            return np.mean(velocities)
        return 0.0

    def calculate_average_temperature(self):
        """Calculates the average temperature of all balls in the simulation."""
        if self.balls:
            temperatures = [ball.calculate_temperature() for ball in self.balls]
            return np.mean(temperatures)
        return 0.0

    def calculate_volume(self):
        """Calculates the volume of the cylindrical well."""
        return np.pi * (self.well.radius ** 2) * self.well.height

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
        ball.position += random_step  # Remove dt scaling

    def update(self):
        """Updates each ball's position, velocity, and handles boundary conditions."""
        num_balls = len(self.balls)

        # Reset forces for all balls
        for ball in self.balls:
            ball.force = np.zeros(3)
            ball.original_velocity = ball.velocity.copy()  # Save the current velocity

        # Compute pairwise repulsion forces
        for i in range(num_balls):
            for j in range(i + 1, num_balls):  # Avoid double-calculating forces
                repulsion_force = self.balls[i].compute_repulsion_force(self.balls[j])
                self.balls[i].force += repulsion_force
                self.balls[j].force -= repulsion_force

        # Update velocity and position for each ball
        for ball in self.balls:
            # Compute acceleration due to forces
            acceleration = ball.force / ball.mass

            # Apply acceleration to velocity
            ball.velocity += acceleration * self.dt

            # Normalize velocity if it exceeds the original value
            original_speed = np.linalg.norm(ball.initial_velocity)
            current_speed = np.linalg.norm(ball.velocity)
            if current_speed > original_speed:
                ball.velocity *= (original_speed / current_speed)

            # Update position
            ball.position += ball.velocity * self.dt

            # Apply bounce and periodic boundary conditions
            ball.velocity = self.well.apply_bounce(ball)
            ball.position, wrapped = self.well.apply_pbc(ball.position)

            # Update path with handling of PBC transition
            ball.skip_path_update = wrapped
            ball.update_path()

    def finalize_simulation(self):
        """Finalizes the path for each ball at the end of the simulation."""
        for ball in self.balls:
            ball.finalize_path()

    def run(self):
        """
        Executes the simulation loop.
        """
        current_time = 0.0
        while current_time < self.total_time:
            self.update()  # Update all ball positions, velocities, and forces
            current_time += self.dt

        # Finalize the simulation by processing results
        self.finalize_simulation()
