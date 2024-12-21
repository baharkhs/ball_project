
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
        if ball.position[2] < wall_decay_length:  # Near bottom wall
            overlap = wall_decay_length - ball.position[2]
            force[2] += repulsion_constant * np.exp(-overlap / wall_decay_length)

        # Repulsion from top z-boundary
        if ball.position[2] > self.height - wall_decay_length:  # Near top wall
            overlap = ball.position[2] - (self.height - wall_decay_length)
            force[2] -= repulsion_constant * np.exp(-overlap / wall_decay_length)

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
        self.velocity = np.array(initial_velocity, dtype=float) if initial_velocity is not None else np.array(
            [0.0, 0.0, 0.0])
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
        # Remove this function entirely.
        pass

    def compute_interaction_force(self, other, interaction_params, box_lengths):
        """
        Compute the Lennard-Jones force between this ball and another, with PBC.

        Args:
            other (Ball): Another ball to calculate the force against.
            interaction_params (dict): Interaction parameters containing epsilon, sigma, and cutoff.
            box_lengths (tuple): Box lengths (x_length, y_length, z_length) for PBC.

        Returns:
            np.array: Force vector in 3D space.
        """
        pair_key = tuple(sorted([self.species, other.species]))
        params = interaction_params.get(pair_key, {"epsilon": 1.0, "sigma": 1.0, "cutoff": 2.5})
        epsilon, sigma, cutoff = params["epsilon"], params["sigma"], params["cutoff"]

        # Calculate displacement using PBC
        delta = self.position - other.position
        delta -= box_lengths * np.round(delta / box_lengths)  # PBC adjustment
        r = np.linalg.norm(delta)

        # Avoid singularities or forces beyond cutoff
        r_min = 0.8 * sigma
        if r < r_min:
            r = r_min
        if r > cutoff:
            return np.zeros(3)

        # Lennard-Jones force calculation
        sr = sigma / r
        sr6 = sr ** 6
        sr12 = sr6 ** 2
        force_magnitude = 24 * epsilon * (2 * sr12 - sr6) / r

        # Cap maximum force for numerical stability
        max_force = 50.0
        force_magnitude = min(force_magnitude, max_force)

        return force_magnitude * (delta / r)

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

    def compute_intra_molecular_forces(self, other, bond_length=1.0, angle=104.5, k_bond=100, k_angle=50):
        """
        Compute harmonic forces to maintain bonds and angles in the molecule.

        Args:
            other (Ball): The second ball in the molecule.
            bond_length (float): Equilibrium bond length.
            angle (float): Equilibrium bond angle in degrees.
            k_bond (float): Bond spring constant.
            k_angle (float): Angle spring constant.

        Returns:
            np.array: Force vector maintaining bonds/angles.
        """
        # Calculate bond distance
        delta = self.position - other.position
        r = np.linalg.norm(delta)

        # Bond force (Hooke's law)
        bond_force = -k_bond * (r - bond_length) * (delta / r)

        # Angle force would require a third atom for angle constraints (to be implemented later)
        return bond_force

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
    def __init__(self, well_radius=0.5, well_height=1.0, total_time=10.0, dt=0.001, movement_type="newtonian"):
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
        self.current_step = 0

        # Define interaction parameters for different species
        self.interaction_params = {
                ("H", "H"): {"epsilon": 1.0, "sigma": 0.1, "cutoff": 2.5},
                ("O", "O"): {"epsilon": 2.0, "sigma": 0.2, "cutoff": 5.0},
                ("H", "O"): {"epsilon": 1.5, "sigma": 0.15, "cutoff": 4.0},
                #("N", "N"): {"epsilon": 1.8, "sigma": 0.12, "cutoff": 3.5},
                #("H", "N"): {"epsilon": 1.2, "sigma": 0.11, "cutoff": 3.0},
                #("O", "N"): {"epsilon": 1.6, "sigma": 0.14, "cutoff": 4.5}
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

    def create_water_molecule(self, center_position, velocity=(0, 0, 0), molecule_id=None):
        """
        Creates a water molecule (H2O) with two hydrogens and one oxygen atom.

        Args:
            center_position (tuple): The position of the oxygen atom.
            velocity (tuple): Initial velocity for all atoms.
            molecule_id (int): Unique molecule ID for tracking intra-molecular forces.
        """
        bond_length = 0.957  # Bond length in angstroms
        angle_deg = 104.5  # Bond angle in degrees
        angle_rad = np.radians(angle_deg)

        # Oxygen atom at the center (Red, Larger Size)
        self.add_ball(
            mass=16.0,
            initial_position=np.array(center_position),
            initial_velocity=np.array(velocity),
            species="O",
            molecule_id=molecule_id
        )
        self.balls[-1].color = "red"
        self.balls[-1].size = 10  # Large size for Oxygen

        # Calculate offsets for hydrogen atoms (Blue, Smaller Size) with Z-component
        offset1 = np.array([bond_length * np.sin(angle_rad / 2), bond_length * np.cos(angle_rad / 2), 0.1])
        offset2 = np.array([-bond_length * np.sin(angle_rad / 2), bond_length * np.cos(angle_rad / 2), -0.1])

        self.add_ball(
            mass=1.0,
            initial_position=np.array(center_position) + offset1,
            initial_velocity=np.array(velocity),
            species="H",
            molecule_id=molecule_id
        )
        self.balls[-1].color = "blue"
        self.balls[-1].size = 6  # Small size for Hydrogen

        self.add_ball(
            mass=1.0,
            initial_position=np.array(center_position) + offset2,
            initial_velocity=np.array(velocity),
            species="H",
            molecule_id=molecule_id
        )
        self.balls[-1].color = "blue"
        self.balls[-1].size = 6  # Small size for Hydrogen

    def apply_monte_carlo_perturbation(self, ball, k_B=0.0083144621, temperature=300):
        """
        Applies a random perturbation to the ball's position and accepts/rejects it using the Metropolis criterion.

        Args:
            ball (Ball): Ball to perturb.
            k_B (float): Boltzmann constant in atomic units.
            temperature (float): System temperature in Kelvin.
        """
        # Store the current position and calculate current energy
        old_position = ball.position.copy()
        old_energy = self.calculate_particle_energy(ball)

        # Propose a random move
        perturbation = np.random.uniform(-0.1, 0.1, size=3)  # Small perturbation in x, y, z
        ball.position += perturbation

        # Calculate the new energy
        new_energy = self.calculate_particle_energy(ball)

        # Metropolis criterion
        delta_energy = new_energy - old_energy
        if delta_energy > 0 and np.random.rand() > np.exp(-delta_energy / (k_B * temperature)):
            ball.position = old_position  # Reject the move

    def calculate_particle_energy(self, ball):
        """
        Calculates the energy of a single particle due to interactions and wall repulsion.

        Args:
            ball (Ball): Ball whose energy will be computed.

        Returns:
            float: The total energy of the particle.
        """
        energy = 0.0
        # Add wall repulsion energy
        energy += np.sum(self.well.compute_wall_repulsion_force(ball) ** 2)

        # Add pairwise interaction energy
        for other in self.balls:
            if other is not ball:
                energy += np.sum(ball.compute_interaction_force(other, self.interaction_params, np.array([2 * self.well.radius, 2 * self.well.radius, self.well.height])) ** 2)
        return energy

    def update(self, rescale_temperature=True, target_temperature=300, rescale_interval=100):
        """
        Updates each ball's position, velocity, and handles forces, boundary conditions,
        temperature monitoring, and intra-molecular bonds.

        Args:
            rescale_temperature (bool): Flag to enable velocity rescaling.
            target_temperature (float): Desired system temperature in Kelvin.
            rescale_interval (int): Frequency (in steps) to apply velocity rescaling.
        """
        num_balls = len(self.balls)
        box_lengths = np.array([2 * self.well.radius, 2 * self.well.radius, self.well.height])

        # Step 1: Reset forces for all balls
        for ball in self.balls:
            ball.force.fill(0)

        # Step 2: Compute pairwise interaction forces
        for i in range(num_balls):
            for j in range(i + 1, num_balls):
                interaction_force = self.balls[i].compute_interaction_force(
                    self.balls[j], self.interaction_params, box_lengths
                )
                self.balls[i].force += interaction_force
                self.balls[j].force -= interaction_force

        # Step 3: Compute bond forces for intra-molecular bonds
        for i in range(num_balls):
            for j in range(i + 1, num_balls):
                if self.balls[i].molecule_id == self.balls[j].molecule_id:  # Same molecule
                    bond_force = self.compute_bond_force(self.balls[i], self.balls[j],
                                                         bond_length=1.0, spring_constant=10.0)
                    self.balls[i].force += bond_force
                    self.balls[j].force -= bond_force

        # Step 4: Compute wall repulsion forces
        for ball in self.balls:
            ball.force += self.well.compute_wall_repulsion_force(ball)

        # Step 5: Update positions and velocities
        for ball in self.balls:
            if self.movement_type == "monte_carlo":
                self.apply_monte_carlo_perturbation(ball)

            # Update velocity and position
            acceleration = ball.force / ball.mass
            ball.velocity += acceleration * self.dt
            ball.position += ball.velocity * self.dt

            # Apply periodic boundary conditions
            self.well.apply_pbc(ball)

            # Update path for visualization
            ball.update_path()

        # Step 6: Apply thermostat (velocity rescaling)
        if rescale_temperature and (self.current_step % rescale_interval == 0):
            self.apply_velocity_rescaling(target_temperature)

        # Step 7: Monitor and collect temperature
        temperature = self.compute_system_temperature()
        print(f"Step {self.current_step}, Temperature: {temperature:.2f} K")
        if not hasattr(self, "temperature_history"):
            self.temperature_history = []
        self.temperature_history.append(temperature)

        self.current_step += 1


    def apply_velocity_rescaling(self, target_temperature):
        """
        Rescales velocities to maintain a target temperature.

        Args:
            target_temperature (float): Desired system temperature in Kelvin.
        """
        k_B = 0.0083144621  # Boltzmann constant in atomic units (amu * (angstrom/fs)^2 / K)

        # Compute the current system temperature from kinetic energy
        total_kinetic_energy = sum(0.5 * ball.mass * np.sum(ball.velocity ** 2) for ball in self.balls)
        num_particles = len(self.balls)

        if num_particles == 0 or total_kinetic_energy == 0:
            return  # Avoid division by zero

        current_temperature = (2 / (3 * num_particles * k_B)) * total_kinetic_energy

        # Compute and apply the scaling factor to rescale velocities
        scaling_factor = np.sqrt(target_temperature / current_temperature)
        for ball in self.balls:
            ball.velocity *= scaling_factor

    def compute_bond_force(self, ball1, ball2, bond_length=1.0, spring_constant=100.0):
        """
        Computes a harmonic bond force to maintain a bond length between two balls.

        Args:
            ball1 (Ball): First ball in the bond.
            ball2 (Ball): Second ball in the bond.
            bond_length (float): Desired bond length.
            spring_constant (float): Strength of the bond constraint.

        Returns:
            np.array: The force vector applied to ball1 (equal and opposite force for ball2).
        """
        delta = ball2.position - ball1.position
        r = np.linalg.norm(delta)

        # Harmonic spring force to maintain bond length
        force_magnitude = -spring_constant * (r - bond_length)
        force_vector = force_magnitude * (delta / r)

        return force_vector

    def compute_system_temperature(self):
        """
        Computes the current temperature of the system based on kinetic energy.

        Returns:
            float: The current system temperature in Kelvin.
        """
        k_B = 0.0083144621  # Boltzmann constant in atomic units

        total_kinetic_energy = sum(0.5 * ball.mass * np.sum(ball.velocity ** 2) for ball in self.balls)
        num_particles = len(self.balls)

        if num_particles == 0:
            return 0.0

        return (2 / (3 * num_particles * k_B)) * total_kinetic_energy

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