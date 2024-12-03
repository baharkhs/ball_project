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

        Args:
            position (np.array): The current position of the particle.

        Returns:
            tuple: (modified position, whether PBC transition occurred)
        """
        wrapped = False
        # Handle PBC in the z-direction
        if position[2] > self.height:  # If above the top boundary
            position[2] -= self.height  # Wrap to the bottom
            wrapped = True
        elif position[2] < 0:  # If below the bottom boundary
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
        # Calculate the distance from the center of the well to the ball's position in the x-y plane
        distance_from_center = np.linalg.norm(ball.position[:2])
        if distance_from_center >= self.radius:  # Check if the ball is beyond or at the boundary
            # Determine the normal direction for reflection
            normal_direction = ball.position[:2] / distance_from_center
            # Reflect the velocity along the normal direction
            ball.velocity[:2] -= 2 * np.dot(ball.velocity[:2], normal_direction) * normal_direction
            # Adjust position to ensure it stays inside the boundary
            ball.position[:2] = normal_direction * (self.radius - 1e-10)  # Small epsilon to keep inside
        return ball.velocity

    def compute_wall_repulsion_force(self, ball, repulsion_constant=1.0):
        """
        Computes a repulsive force between a ball and the well walls (x-y boundary and z boundary).

        Args:
            ball (Ball): The ball object.
            repulsion_constant (float): Strength of the repulsion force.

        Returns:
            np.array: The repulsion force vector acting on the ball.
        """
        force = np.zeros(3)

        # Compute repulsion from the cylindrical x-y boundary
        distance_from_center = np.linalg.norm(ball.position[:2])
        if distance_from_center >= self.radius - ball.radius:  # Check if near or beyond the boundary
            overlap = ball.radius - (self.radius - distance_from_center)  # Penetration depth
            if overlap > 0:
                # Compute the normal vector and apply repulsion
                normal_direction = ball.position[:2] / distance_from_center
                repulsion_magnitude = repulsion_constant * overlap
                force[:2] = -repulsion_magnitude * normal_direction  # Push inward

        # Compute repulsion from the bottom z boundary
        if ball.position[2] < ball.radius:  # Near the bottom wall
            overlap = ball.radius - ball.position[2]
            if overlap > 0:
                force[2] += repulsion_constant * overlap  # Push upward

        # Compute repulsion from the top z boundary
        elif ball.position[2] > self.height - ball.radius:  # Near the top wall
            overlap = ball.radius - (self.height - ball.position[2])
            if overlap > 0:
                force[2] -= repulsion_constant * overlap  # Push downward

        return force




class Ball:
    def __init__(self, mass=18.0, initial_position=None, initial_velocity=None, species="O", molecule_id=None):
        """
        Represents a single particle with mass, position, velocity, and temperature attributes.

        Args:
            mass (float): Mass of the particle in atomic mass units (amu).
            initial_position (array-like): Initial [x, y, z] position in angstroms.
            initial_velocity (array-like): Initial [vx, vy, vz] velocity in angstroms per femtosecond (fs).
        """
        self.mass = mass  # Mass in atomic mass units (amu).
        self.position = np.array(initial_position) if initial_position is not None else np.array([0.0, 0.0, 0.0])
        self.velocity = np.array(initial_velocity, dtype=float) if initial_velocity is not None else np.array([0.0, 0.0, 0.0])
        self.initial_velocity = self.velocity.copy()  # Store the initial velocity for reference or reuse.
        self.path_segments = []  # Stores completed path segments for visualization.
        self.current_path_segment = {"x": [], "y": [], "z": []}  # Tracks the ongoing path segment.
        self.skip_path_update = False  # Flag to indicate if path updates should be skipped (e.g., during PBC transitions).
        self.temperature = self.calculate_temperature()  # Calculate the initial temperature of the ball.
        self.force = np.zeros(3)  # Initialize the force acting on the ball to zero.
        self.radius = 0.1  # Radius of the ball, used for collision detection or wall interactions.
        self.species = species  # Type of particle ("H", "O")
        self.molecule_id = molecule_id  # Assign a molecule ID for intra/inter-molecular differentiation.

    def compute_repulsion_force(self, other, repulsion_constant=1.0):
        """
        Compute the repulsive force exerted by another ball.

        Args:
            other (Ball): The other ball in the simulation.
            repulsion_constant (float): Constant controlling the strength of repulsion.

        Returns:
            np.array: The repulsive force vector acting on the ball.
        """
        displacement = self.position - other.position  # Vector from the other ball to this ball.
        distance = np.linalg.norm(displacement)  # Compute the distance between the two balls.
        if distance == 0 or distance > 5 * self.radius:  # Ignore if the balls are too far apart or overlapping.
            return np.zeros(3)

        # Cap the repulsion force to avoid excessive acceleration.
        max_force = 10.0
        force_magnitude = min(repulsion_constant / (distance ** 2), max_force)  # Inverse-square law for repulsion.
        force_direction = displacement / distance  # Normalize the displacement vector to get direction.
        return force_magnitude * force_direction  # Return the force vector.

    def compute_interaction_force(self, other, interaction_params):
        """
        Compute the interaction force between this ball and another ball based on their types.

        Args:
            other (Ball): The other ball in the simulation.
            interaction_params (dict): A dictionary containing force parameters for different particle pairs.

        Returns:
            np.array: The interaction force vector acting on this ball.
        """
        pair_key = tuple(sorted([self.species, other.species]))  # Determine the interaction type (e.g., ("H", "O"))
        params = interaction_params.get(pair_key, {})

        displacement = self.position - other.position
        distance = np.linalg.norm(displacement)

        if distance == 0 or distance > params.get("cutoff", 5.0):  # Ignore if too far apart or overlapping.
            return np.zeros(3)

        # Compute attraction and repulsion forces
        repulsion_strength = params.get("repulsion", 1.0) / (distance ** 2)
        attraction_strength = params.get("attraction", 0.1) * (1 / distance)
        force_magnitude = repulsion_strength - attraction_strength

        force_direction = displacement / distance
        return force_magnitude * force_direction

    def calculate_kinetic_energy(self):
        """
        Calculates and returns the kinetic energy (KE) of the ball.

        Returns:
            float: Kinetic energy in atomic units (amu * (angstrom/fs)^2).
        """
        return 0.5 * self.mass * np.dot(self.velocity, self.velocity)  # KE = 0.5 * m * v^2.

    def calculate_temperature(self):
        """
        Calculates the temperature of the ball based on its kinetic energy.

        Returns:
            float: Temperature in Kelvin (K).
        """
        k_B = 0.0083144621  # Boltzmann constant in atomic units (amu * (angstrom/fs)^2 / K).
        kinetic_energy = self.calculate_kinetic_energy()  # Compute kinetic energy.
        return (2 / 3) * (kinetic_energy / k_B)  # Relate KE to temperature (classical thermodynamics).

    def apply_forces(self):
        """
        Placeholder for applying external forces (e.g., gravity) to the ball.
        """
        # Extend this method in the future to include external forces (e.g., gravity or electric fields).
        pass

    def update_velocity_position(self, dt):
        """
        Updates the velocity and position of the ball based on the time step.

        Args:
            dt (float): Time step in femtoseconds.
        """
        # Update the position using velocity (Newtonian motion: x = x + v * dt).
        self.position += self.velocity * dt

    def update_path(self):
        """
        Records the ball's position for path visualization. Handles PBC transitions
        and avoids adding positions exactly at PBC boundaries to the path.
        """
        if self.skip_path_update:
            # If a PBC transition occurred, finalize the current segment and start a new one.
            if self.current_path_segment["x"]:
                self.path_segments.append(self.current_path_segment)
            self.current_path_segment = {"x": [], "y": [], "z": []}
            self.skip_path_update = False  # Reset the flag for future updates.
        else:
            # Skip recording the path if the ball is exactly at the PBC boundary.
            if self.position[2] == 0 or self.position[2] == self.radius:
                return

            # Otherwise, update the ongoing path segment.
            self.current_path_segment["x"].append(self.position[0])
            self.current_path_segment["y"].append(self.position[1])
            self.current_path_segment["z"].append(self.position[2])

    def finalize_path(self):
        """
        Finalizes and stores the current path segment at the end of the simulation.
        """
        if self.current_path_segment["x"]:  # Check if the current path segment contains data.
            self.path_segments.append(self.current_path_segment)  # Store the segment.

    def get_path_segments(self):
        """
        Returns all path segments for visualization.

        Returns:
            list: A list of all recorded path segments, including the current one.
        """
        return self.path_segments + [self.current_path_segment]  # Include the current segment.

    def finalize_simulation(self):
        """
        Finalizes the path for the ball at the end of the simulation.
        """
        self.finalize_path()  # Ensure the current path segment is stored.



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
        self.well = Well(well_radius, well_height)  # Initialize the cylindrical well.
        self.dt = dt  # Time step for each simulation iteration.
        self.total_time = total_time  # Total simulation time.
        self.balls = []  # A list to hold all ball objects in the simulation.
        self.movement_type = movement_type  # Type of movement for the simulation: "newtonian" or "monte_carlo".

        # Define interaction parameters for different species
        self.interaction_params = {
            ("H", "H"): {"repulsion": 1.0, "attraction": 0.05, "cutoff": 3.0},
            ("O", "O"): {"repulsion": 2.0, "attraction": 0.01, "cutoff": 5.0},
            ("H", "O"): {"repulsion": 1.5, "attraction": 0.1, "cutoff": 4.0},
        }

    def set_movement_type(self, movement_type="newtonian"):
        """Sets the movement type for all balls in the simulation."""
        self.movement_type = movement_type

    def add_ball(self, mass=1.0, initial_position=None, initial_velocity=None, species="O", molecule_id=None):
        """
        Adds a ball to the simulation with the specified properties.

        Args:
            mass (float): Mass of the particle.
            initial_position (array-like): Initial position of the particle.
            initial_velocity (array-like): Initial velocity of the particle.
            species (str): Type of particle (e.g., "H" for hydrogen, "O" for oxygen).
            molecule_id (int): Molecule identifier for intra/inter-molecular interaction.
        """
        ball = Ball(mass=mass, initial_position=initial_position, initial_velocity=initial_velocity,
                    species=species, molecule_id=molecule_id)
        self.balls.append(ball)

        

    def apply_monte_carlo_perturbation(self, ball):
        """
        Applies random noise to the force of a ball, simulating stochastic Monte Carlo behavior.

        Args:
            ball (Ball): Ball to perturb.
        """
        noise = np.random.uniform(-0.1, 0.1, size=3)  # Random noise in x, y, z directions.
        ball.force += noise

    def update(self):
        """
        Updates each ball's position, velocity, and handles boundary conditions.
        """
        num_balls = len(self.balls)  # Get the total number of balls in the simulation.

        # Reset forces for all balls
        for ball in self.balls:
            ball.force = np.zeros(3)  # Reset the force vector for each ball to zero.

        # Compute pairwise interaction forces between balls
        for i in range(num_balls):
            for j in range(i + 1, num_balls):  # Avoid double-calculating forces.
                interaction_force = self.balls[i].compute_interaction_force(
                    self.balls[j], self.interaction_params
                )
                self.balls[i].force += interaction_force  # Add the interaction force to ball i.
                self.balls[j].force -= interaction_force  # Apply equal and opposite force to ball j.

        # Compute wall repulsion forces
        for ball in self.balls:
            wall_repulsion_force = self.well.compute_wall_repulsion_force(ball)  # Calculate force from the wall.
            ball.force += wall_repulsion_force  # Add the wall repulsion force to the ball's total force.

            # Apply Monte Carlo perturbation in forces if enabled
            if self.movement_type == "monte_carlo":
                self.apply_monte_carlo_perturbation(ball)

        # Update velocity and position for each ball
        for ball in self.balls:
            if self.movement_type == "monte_carlo":
                # Apply Monte Carlo movement
                self.apply_monte_carlo_movement(ball)
            else:
                # Apply acceleration for Newtonian movement
                acceleration = ball.force / ball.mass  # Calculate acceleration using F = ma.
                ball.velocity += acceleration * self.dt  # Update velocity using v = u + at.
                ball.position += ball.velocity * self.dt  # Update position using x = x + vt.

                # Apply gradual velocity decay (keep for future use)
                decay_factor = 1
                ball.velocity = decay_factor * ball.velocity  # Reduce velocity for damping effects.

            # Apply bounce and periodic boundary conditions
            ball.velocity = self.well.apply_bounce(ball)  # Check and handle bounces on the cylindrical wall.
            ball.position, wrapped = self.well.apply_pbc(ball.position)  # Handle PBC on the z-axis.

            # Handle PBC transition: defer adding wrapped position to path
            if wrapped:
                ball.skip_path_update = True  # Mark PBC transition.
            else:
                ball.skip_path_update = False  # Reset if no transition.

            # Update path (skip wrapped positions)
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
            self.update()  # Update all ball positions, velocities, and forces.
            current_time += self.dt  # Increment the current time by the time step.

        # Finalize the simulation by processing results.
        self.finalize_simulation()




