import numpy as np

class Well:
    def __init__(self, radius=5.0, height=10.0, cutoff_radius=None):
        """
        Initializes the Well object representing a cylindrical space for particle movement.

        Args:
            radius (float): Radius of the well in angstroms (1e-10 meters).
            height (float): Height of the well in angstroms.
            cutoff_radius (float): Optional cutoff radius for smoothing forces (default: radius).
        """
        self.radius = radius  # Radius of the cylindrical boundary
        self.height = height  # Height of the well (z-axis boundary)
        self.cutoff_radius = cutoff_radius if cutoff_radius else radius  # Smooth cutoff radius

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
        distance_from_center = np.linalg.norm(ball.position[:2])  # Distance from center in x-y plane
        if distance_from_center >= self.radius:  # If at or beyond the boundary
            normal_direction = ball.position[:2] / distance_from_center  # Normal vector
            ball.velocity[:2] -= 2 * np.dot(ball.velocity[:2], normal_direction) * normal_direction  # Reflect velocity
            ball.position[:2] = normal_direction * (self.radius - 1e-10)  # Adjust position to stay inside
        return ball.velocity

    def smooth_cutoff_function(self, r, cutoff):
        """
        Computes a smooth cutoff factor to taper forces near the cutoff radius.

        Args:
            r (float): Distance of the particle from the boundary.
            cutoff (float): Cutoff distance.

        Returns:
            float: A factor to scale the force smoothly to zero at the cutoff.
        """
        if r > cutoff:
            return 0.0  # Beyond the cutoff, force is zero
        # Smoothly decay to zero at cutoff
        return (1 - (r / cutoff)**2)**2

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

        # Repulsion from the cylindrical x-y boundary
        distance_from_center = np.linalg.norm(ball.position[:2])
        if distance_from_center >= self.radius - ball.radius:  # Near or beyond boundary
            overlap = ball.radius - (self.radius - distance_from_center)  # Penetration depth
            if overlap > 0:
                normal_direction = ball.position[:2] / distance_from_center  # Normal vector
                repulsion_magnitude = repulsion_constant * overlap
                smooth_factor = self.smooth_cutoff_function(distance_from_center, self.cutoff_radius)  # Smooth decay
                force[:2] = -repulsion_magnitude * smooth_factor * normal_direction  # Apply smooth repulsion

        # Repulsion from the bottom z boundary
        if ball.position[2] < ball.radius:  # Near the bottom wall
            overlap = ball.radius - ball.position[2]
            if overlap > 0:
                smooth_factor = self.smooth_cutoff_function(ball.position[2], ball.radius)  # Smooth decay
                force[2] += repulsion_constant * overlap * smooth_factor  # Push upward

        # Repulsion from the top z boundary
        elif ball.position[2] > self.height - ball.radius:  # Near the top wall
            overlap = ball.radius - (self.height - ball.position[2])
            if overlap > 0:
                smooth_factor = self.smooth_cutoff_function(self.height - ball.position[2], ball.radius)  # Smooth decay
                force[2] -= repulsion_constant * overlap * smooth_factor  # Push downward

        return force

    def compute_total_wall_force(self, ball, repulsion_constant=1.0):
        """
        Combines the wall repulsion forces and periodic boundary conditions for more robust handling.

        Args:
            ball (Ball): The ball object.
            repulsion_constant (float): Strength of the repulsion force.

        Returns:
            np.array: Total force acting on the ball from the well boundaries.
        """
        repulsion_force = self.compute_wall_repulsion_force(ball, repulsion_constant)
        # Additional wall-related forces can be added here if necessary
        return repulsion_force




class Ball:
    def __init__(self, mass=18.0, initial_position=None, initial_velocity=None, species="O", molecule_id=None):
        """
        Represents a single particle with mass, position, velocity, and temperature attributes.

        Args:
            mass (float): Mass of the particle in atomic mass units (amu).
            initial_position (array-like): Initial [x, y, z] position in angstroms.
            initial_velocity (array-like): Initial [vx, vy, vz] velocity in angstroms per femtosecond (fs).
            species (str): Particle type ("H" for hydrogen, "O" for oxygen, etc.).
            molecule_id (int): Identifier for molecule (for intra/inter-molecular forces).
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
        self.radius = 0.1 if species == "H" else 0.2  # Assign radius based on species.
        self.species = species  # Type of particle ("H", "O", etc.).
        self.molecule_id = molecule_id  # Assign a molecule ID for intra/inter-molecular differentiation.
        self.potential_energy = 0.0  # Placeholder for potential energy calculations.
        self.lennard_jones_params = {"epsilon": 0.1, "sigma": 0.3}  # Default Lennard-Jones parameters.
        self.rescaling_factor = 1.0  # Temperature rescaling factor (to adjust velocities if needed).

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
        Compute the interaction force between this ball and another ball using Lennard-Jones potential.

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

        # Lennard-Jones potential-based interaction forces
        sigma = params.get("sigma", self.lennard_jones_params["sigma"])
        epsilon = params.get("epsilon", self.lennard_jones_params["epsilon"])

        # Lennard-Jones force computation
        force_magnitude = 48 * epsilon * (sigma ** 12 / distance ** 13 - 0.5 * sigma ** 6 / distance ** 7)
        force_direction = displacement / distance
        return force_magnitude * force_direction

    def calculate_potential_energy(self, other, interaction_params):
        """
        Calculates Lennard-Jones potential energy with another ball.

        Args:
            other (Ball): The other ball in the simulation.
            interaction_params (dict): Interaction parameters for Lennard-Jones potential.

        Returns:
            float: Potential energy due to interaction with another particle.
        """
        pair_key = tuple(sorted([self.species, other.species]))
        params = interaction_params.get(pair_key, {})
        sigma = params.get("sigma", self.lennard_jones_params["sigma"])
        epsilon = params.get("epsilon", self.lennard_jones_params["epsilon"])
        displacement = self.position - other.position
        distance = np.linalg.norm(displacement)
        if distance == 0 or distance > params.get("cutoff", 5.0):
            return 0.0
        return 4 * epsilon * ((sigma / distance) ** 12 - (sigma / distance) ** 6)

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

    def rescale_velocity(self, target_temperature):
        """
        Rescales the velocity to match a target temperature.

        Args:
            target_temperature (float): The desired target temperature in Kelvin.
        """
        current_temperature = self.calculate_temperature()
        if current_temperature > 0:  # Avoid division by zero
            scaling_factor = np.sqrt(target_temperature / current_temperature)
            self.velocity *= scaling_factor
            self.rescaling_factor = scaling_factor  # Store the rescaling factor

    def update_velocity_position(self, dt):
        """
        Updates the velocity and position of the ball based on the time step.

        Args:
            dt (float): Time step in femtoseconds.
        """
        acceleration = self.force / self.mass  # Acceleration from force
        self.velocity += acceleration * dt  # Update velocity
        self.position += self.velocity * dt  # Update position

    def update_path(self):
        """
        Records the ball's position for path visualization. Handles PBC transitions
        and avoids adding positions exactly at PBC boundaries to the path.
        """
        if self.skip_path_update:
            if self.current_path_segment["x"]:
                self.path_segments.append(self.current_path_segment)
            self.current_path_segment = {"x": [], "y": [], "z": []}
            self.skip_path_update = False  # Reset flag
        else:
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
        self.average_temperature = 0.0  # Initialize average temperature.
        self.total_energy = 0.0  # Initialize total energy.

        # Define interaction parameters for different species based on Lennard-Jones potential
        self.interaction_params = {
            ("O", "O"): {"epsilon": 0.5, "sigma": 0.3, "cutoff": 5.0},
            ("H", "O"): {"epsilon": 0.3, "sigma": 0.25, "cutoff": 4.0},
            ("H", "H"): {"epsilon": 0.2, "sigma": 0.2, "cutoff": 3.0},
        }

    def set_movement_type(self, movement_type="newtonian"):
        """
        Sets the movement type for the simulation.

        Args:
            movement_type (str): Type of movement, either "newtonian" or "monte_carlo".
        """
        self.movement_type = movement_type

    def add_ball(self, mass=1.0, initial_position=None, initial_velocity=None, species="O", molecule_id=None):
        """
        Adds a ball to the simulation with the specified properties.

        Args:
            mass (float): Mass of the particle in atomic mass units (amu).
            initial_position (array-like): Initial position of the particle in angstroms.
            initial_velocity (array-like): Initial velocity of the particle in angstroms/fs.
            species (str): Type of particle (e.g., "H" for hydrogen, "O" for oxygen).
            molecule_id (int): Molecule identifier for intra/inter-molecular interactions.
        """
        ball = Ball(
            mass=mass,
            initial_position=initial_position,
            initial_velocity=initial_velocity,
            species=species,
            molecule_id=molecule_id,
        )
        self.balls.append(ball)

    def apply_monte_carlo_movement(self, ball):
        """
        Applies a Monte Carlo random walk to the ball's position.
        This introduces stochastic positional changes independent of force calculations.

        Args:
            ball (Ball): The particle to update.

        Updates:
            ball.position: Adds a random step to the ball's position in x, y, z.
        """
        max_step_size = 0.01  # Maximum random step size in each dimension (angstroms).
        random_step = np.random.uniform(-max_step_size, max_step_size, size=3)  # Random displacement
        ball.position += random_step  # Update position

    def apply_monte_carlo_perturbation(self, ball: Ball):
        """
        Adds random noise to the force acting on a ball, simulating stochastic forces in Monte Carlo dynamics.

        Args:
            ball (Ball): The ball to which the noise is applied.
        """
        noise_magnitude = 0.1  # Maximum magnitude of stochastic force perturbation
        noise = np.random.uniform(-noise_magnitude, noise_magnitude, size=3)  # Generate random noise
        ball.force += noise  # Apply noise to the force

    def rescale_temperatures(self, target_temperature):
        for ball in self.balls:
            current_temp = ball.calculate_temperature()
            scaling_factor = np.sqrt(target_temperature / current_temp) if current_temp > 0 else 1.0
            print(
                f"Rescaling {ball.species} ball: Current Temp = {current_temp:.2f}, Scaling Factor = {scaling_factor:.2f}")
            ball.velocity *= scaling_factor

    def update(self):
        """
        Updates the positions, velocities, and forces acting on all balls in the simulation.
        """
        # Step 1: Reset forces and calculate average temperature
        self.average_temperature = self.calculate_average_temperature()
        for ball in self.balls:
            ball.force = np.zeros(3)  # Reset to zero before force calculations

        # Step 2: Calculate pairwise interaction forces
        num_balls = len(self.balls)
        for i in range(num_balls):
            for j in range(i + 1, num_balls):  # Avoid double counting
                interaction_force = self.balls[i].compute_interaction_force(
                    self.balls[j], self.interaction_params
                )
                self.balls[i].force += interaction_force
                self.balls[j].force -= interaction_force

        # Step 3: Calculate wall repulsion forces
        for ball in self.balls:
            wall_force = self.well.compute_wall_repulsion_force(ball)
            ball.force += wall_force

            if self.movement_type == "monte_carlo":
                self.apply_monte_carlo_perturbation(ball)

        # Step 4: Update ball velocities and positions
        for ball in self.balls:
            if self.movement_type == "monte_carlo":
                self.apply_monte_carlo_movement(ball)
            else:
                ball.update_velocity_position(self.dt)

            # Apply boundary conditions
            ball.velocity = self.well.apply_bounce(ball)
            ball.position, wrapped = self.well.apply_pbc(ball.position)

            if wrapped:
                ball.skip_path_update = True
            else:
                ball.skip_path_update = False

            ball.update_path()

        # Step 5: Calculate total energy of the system
        self.total_energy = self.calculate_total_energy()

        for ball in self.balls:
            print(f"Ball {ball.species}: Force = {ball.force}, Velocity = {ball.velocity}")

    def calculate_average_temperature(self):
        """
        Calculates the average temperature of all balls in the simulation.

        Returns:
            float: Average temperature in Kelvin.
        """
        if not self.balls:
            return 0.0  # Avoid division by zero
        temperatures = [ball.calculate_temperature() for ball in self.balls]
        return sum(temperatures) / len(temperatures)

    def calculate_total_energy(self):
        """
        Calculates the total energy (kinetic + potential) of the system.

        Returns:
            float: Total energy in atomic units.
        """
        total_kinetic = sum(ball.calculate_kinetic_energy() for ball in self.balls)
        total_potential = sum(
            self.balls[i].calculate_potential_energy(self.balls[j], self.interaction_params)
            for i in range(len(self.balls)) for j in range(i + 1, len(self.balls))
        )
        return total_kinetic + total_potential

    def finalize_simulation(self):
        """
        Finalizes the simulation by ensuring that all paths are stored and any required cleanup is done.
        """
        for ball in self.balls:
            ball.finalize_path()

    def run(self, target_temperature=None):
        """
        Runs the simulation loop over the specified total simulation time.

        This method updates ball positions, velocities, and forces at each time step,
        and finalizes the results at the end.

        Args:
            target_temperature (float): If provided, rescales temperatures periodically to maintain equilibrium.
        """
        current_time = 0.0
        rescale_interval = 10  # Rescale every 10 steps for finer control
        step_count = 0

        while current_time < self.total_time:
            self.update()  # Update all ball properties for the current time step

            # Rescale velocities to maintain target temperature if provided
            if target_temperature and step_count % rescale_interval == 0:
                print(f"Rescaling velocities at step {step_count} to target temperature {target_temperature} K")
                self.rescale_temperatures(target_temperature)

            # Debugging: Print the average temperature at every step
            print(f"Step {step_count}: Average Temperature = {self.average_temperature:.2f} K")

            current_time += self.dt
            step_count += 1

        # Finalize results
        self.finalize_simulation()



