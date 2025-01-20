
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


class Ball:
    def __init__(self, mass=18.0, initial_position=None, initial_velocity=None, species="O", molecule_id=None,
                 color=None, size=None):
        """
        Represents a single particle with mass, position, velocity, and attributes.

        Args:
            mass (float): Mass of the particle in atomic mass units (amu).
            initial_position (array-like): Initial [x, y, z] position in angstroms.
            initial_velocity (array-like): Initial [vx, vy, vz] velocity in angstroms per femtosecond (fs).
            species (str): Type of particle ("H" for Hydrogen, "O" for Oxygen).
            molecule_id (int): ID for tracking molecules.
            color (str): Visualization color (optional).
            size (float): Visualization size (optional).
        """
        self.mass = mass  # Mass in atomic mass units (amu).
        self.position = np.array(initial_position) if initial_position is not None else np.array([0.0, 0.0, 0.0])
        self.velocity = np.array(initial_velocity, dtype=float) if initial_velocity is not None else np.array(
            [0.0, 0.0, 0.0])
        self.initial_velocity = self.velocity.copy()
        self.path_segments = []
        self.current_path_segment = {"x": [], "y": [], "z": []}
        self.skip_path_update = False
        self.temperature = self.calculate_temperature()
        self.force = np.zeros(3)
        self.radius = 0.1
        self.species = species
        self.molecule_id = molecule_id

        # Assign colors based on species, allowing overrides
        self.color = color if color else ("red" if species == "O" else "blue" if species == "H" else "gray")

        # Assign default size with overrides
        self.size = size if size else (10 if species == "O" else 6 if species == "H" else 8)

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

    @staticmethod
    def lennard_jones_potential(r, epsilon, sigma):
        """
        Calculate the Lennard-Jones potential for a given distance.

        Args:
            r (float): Distance between particles.
            epsilon (float): Depth of the potential well.
            sigma (float): Distance at which the potential is zero.

        Returns:
            float: Potential energy at distance r.
        """
        if r == 0:
            return np.inf  # Avoid division by zero
        sr = sigma / r
        sr6 = sr ** 6
        sr12 = sr6 ** 2
        return 4 * epsilon * (sr12 - sr6)

    @staticmethod
    def lennard_jones_force(r, epsilon, sigma):
        """
        Calculate the Lennard-Jones force for a given distance.

        Args:
            r (float): Distance between particles.
            epsilon (float): Depth of the potential well.
            sigma (float): Distance at which the potential is zero.

        Returns:
            float: Force magnitude at distance r.
        """
        if r == 0:
            return np.inf  # Avoid division by zero
        sr = sigma / r
        sr6 = sr ** 6
        sr12 = sr6 ** 2
        force_magnitude = 24 * epsilon * (2 * sr12 - sr6) / r
        return force_magnitude



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
        self.potential_energy_data = []  # To store (distance, potential_energy) pairs
        self.temperature_history = []  # Store temperature data
        self.collective_variables = {
            "total_energy": [],
            "radial_distribution": []
        }

        # Define interaction parameters for different species
        self.interaction_params = {
            ("H", "H"): {"epsilon": 1.0, "sigma": 0.1, "cutoff": 2.5},
            ("O", "O"): {"epsilon": 2.0, "sigma": 0.2, "cutoff": 5.0},
            ("H", "O"): {"epsilon": 1.5, "sigma": 0.15, "cutoff": 4.0}
        }

    def set_movement_type(self, movement_type="newtonian"):
        """Sets the movement type for all balls in the simulation."""
        self.movement_type = movement_type

    def add_ball(self, mass=1.0, initial_position=None, initial_velocity=None, species="O", molecule_id=None,
                 color=None, size=None):
        """
        Adds a ball to the simulation with the specified properties.

        Args:
            mass (float): Mass of the particle in atomic mass units (amu).
            initial_position (array-like): Initial [x, y, z] position in angstroms.
            initial_velocity (array-like): Initial [vx, vy, vz] velocity in angstroms per femtosecond (fs).
            species (str): Type of particle ("O", "H").
            molecule_id (int): Unique identifier for molecules.
            color (str): Color for visualization.
            size (int): Size of the ball for visualization.
        """

        if initial_position is None:
            initial_position = [len(self.balls) * 0.1, len(self.balls) * 0.1, len(self.balls) * 0.1]

        ball = Ball(mass=mass, initial_position=initial_position, initial_velocity=initial_velocity, species=species,
                    molecule_id=molecule_id)

        # Assign color and size based on species or provided values
        if color is None or size is None:
            if species == "O":
                ball.color = "red"
                ball.size = 10  # Oxygen should have a larger size
            elif species == "H":
                ball.color = "blue"
                ball.size = 6  # Hydrogen should have a smaller size
            else:
                ball.color = "gray"
                ball.size = 8  # Default for other atoms
        else:
            ball.color = color
            ball.size = size

        self.balls.append(ball)

    def create_water_molecule(self, center_position, velocity=(0, 0, 0), molecule_id=None):
        """
        Creates a water molecule (H2O) with two hydrogens and one oxygen atom.
        """
        bond_length = 0.957
        angle_deg = 104.5
        angle_rad = np.radians(angle_deg)

        self.add_ball(mass=16.0, initial_position=np.array(center_position), initial_velocity=np.array(velocity), species="O", molecule_id=molecule_id)
        offset1 = np.array([bond_length * np.sin(angle_rad / 2), bond_length * np.cos(angle_rad / 2), 0.1])
        offset2 = np.array([-bond_length * np.sin(angle_rad / 2), bond_length * np.cos(angle_rad / 2), -0.1])

        self.add_ball(mass=1.0, initial_position=np.array(center_position) + offset1, initial_velocity=np.array(velocity), species="H", molecule_id=molecule_id)
        self.add_ball(mass=1.0, initial_position=np.array(center_position) + offset2, initial_velocity=np.array(velocity), species="H", molecule_id=molecule_id)

    def apply_monte_carlo_perturbation(self, ball, k_B=0.0083144621, temperature=300):
        """
        Applies a random perturbation to the ball's position and accepts/rejects it using the Metropolis criterion.
        """
        old_position = ball.position.copy()
        old_energy = self.calculate_particle_energy(ball)

        perturbation = np.random.uniform(-0.1, 0.1, size=3)
        ball.position += perturbation

        new_energy = self.calculate_particle_energy(ball)

        delta_energy = new_energy - old_energy
        if delta_energy > 0 and np.random.rand() > np.exp(-delta_energy / (k_B * temperature)):
            ball.position = old_position

    def calculate_particle_energy(self, ball):
        """
        Calculates the energy of a single particle due to interactions and wall repulsion.
        """
        energy = 0.0
        energy += np.sum(self.well.compute_wall_repulsion_force(ball) ** 2)

        for other in self.balls:
            if other is not ball:
                energy += np.sum(ball.compute_interaction_force(other, self.interaction_params, np.array([2 * self.well.radius, 2 * self.well.radius, self.well.height])) ** 2)
        return energy

    def compute_potential_energy_data(self):
        """
        Compute potential energy vs. distance for all particle pairs.
        """
        distances = []
        potential_energies = []
        num_balls = len(self.balls)
        box_lengths = np.array([2 * self.well.radius, 2 * self.well.radius, self.well.height])

        for i in range(num_balls):
            for j in range(i + 1, num_balls):
                ball1, ball2 = self.balls[i], self.balls[j]

                delta = ball1.position - ball2.position
                delta -= box_lengths * np.round(delta / box_lengths)
                r = np.linalg.norm(delta)

                pair_key = tuple(sorted([ball1.species, ball2.species]))
                params = self.interaction_params.get(pair_key, {"epsilon": 1.0, "sigma": 1.0})
                epsilon, sigma = params["epsilon"], params["sigma"]

                potential_energy = Ball.lennard_jones_potential(r, epsilon, sigma)

                distances.append(r)
                potential_energies.append(potential_energy)

        return distances, potential_energies

    def compute_radial_distribution_function(self, bins=50):
        """
        Computes the radial distribution function (RDF) for the particles.
        """
        num_balls = len(self.balls)
        box_lengths = np.array([2 * self.well.radius, 2 * self.well.radius, self.well.height])
        distances = []

        for i in range(num_balls):
            for j in range(i + 1, num_balls):
                delta = self.balls[i].position - self.balls[j].position
                delta -= box_lengths * np.round(delta / box_lengths)
                distances.append(np.linalg.norm(delta))

        hist, bin_edges = np.histogram(distances, bins=bins, density=True)
        self.collective_variables["radial_distribution"].append((bin_edges[:-1], hist))

    def update(self, rescale_temperature=True, target_temperature=300, rescale_interval=100):
        """
        Updates each ball's position, velocity, and handles forces, boundary conditions, and temperature monitoring.
        """
        if not self.balls:
            print("No balls in the simulation to update.")
            return

        num_balls = len(self.balls)
        box_lengths = np.array([2 * self.well.radius, 2 * self.well.radius, self.well.height])

        for ball in self.balls:
            ball.force.fill(0)

        for i in range(num_balls):
            for j in range(i + 1, num_balls):
                interaction_force = self.balls[i].compute_interaction_force(self.balls[j], self.interaction_params, box_lengths)
                self.balls[i].force += interaction_force
                self.balls[j].force -= interaction_force

        for ball in self.balls:
            ball.force += self.well.compute_wall_repulsion_force(ball)

        for ball in self.balls:
            if self.movement_type == "monte_carlo":
                self.apply_monte_carlo_perturbation(ball)

            acceleration = ball.force / ball.mass
            ball.velocity += acceleration * self.dt
            ball.position += ball.velocity * self.dt

            self.well.apply_pbc(ball)
            ball.update_path()

        if rescale_temperature and (self.current_step % rescale_interval == 0):
            self.apply_velocity_rescaling(target_temperature)

        temperature = self.compute_system_temperature()
        print(f"Step {self.current_step}, Temperature: {temperature:.2f} K")
        self.temperature_history.append(temperature)

        self.current_step += 1

        total_energy = sum(self.calculate_particle_energy(ball) for ball in self.balls)
        self.collective_variables["total_energy"].append(total_energy)
        self.compute_radial_distribution_function()

    def apply_velocity_rescaling(self, target_temperature):
        """
        Rescales velocities to maintain a target temperature.
        """
        k_B = 0.0083144621
        total_kinetic_energy = sum(0.5 * ball.mass * np.sum(ball.velocity ** 2) for ball in self.balls)
        num_particles = len(self.balls)

        if num_particles == 0 or total_kinetic_energy == 0:
            return

        current_temperature = (2 / (3 * num_particles * k_B)) * total_kinetic_energy
        scaling_factor = np.sqrt(target_temperature / current_temperature)
        for ball in self.balls:
            ball.velocity *= scaling_factor

    def compute_system_temperature(self):
        """
        Computes the current temperature of the system based on kinetic energy.
        """
        k_B = 0.0083144621
        total_kinetic_energy = sum(0.5 * ball.mass * np.sum(ball.velocity ** 2) for ball in self.balls)
        num_particles = len(self.balls)

        if num_particles == 0:
            return 0.0

        return (2 / (3 * num_particles * k_B)) * total_kinetic_energy

    def plot_potential_energy(self):
        """
        Plot potential energy vs. distance for the simulation.
        """
        distances, potential_energies = self.compute_potential_energy_data()

        if self.balls:
            pair_key = tuple(sorted([self.balls[0].species, self.balls[1].species]))
            params = self.interaction_params.get(pair_key, {"epsilon": 1.0, "sigma": 1.0})
            epsilon, sigma = params["epsilon"], params["sigma"]
        else:
            epsilon, sigma = 1.0, 1.0

        r_theoretical = np.linspace(0.5, 2.5, 100)
        theoretical_potential = [Ball.lennard_jones_potential(r, epsilon, sigma) for r in r_theoretical]

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))

        plt.plot(r_theoretical, theoretical_potential, color='red',
                 label=fr"Theoretical LJ ($\epsilon$={epsilon}, $\sigma$={sigma})")

        plt.scatter(distances, potential_energies, color="blue", alpha=0.6, label="Simulation Data")

        plt.xlabel("Distance (Å)")
        plt.ylabel("Potential Energy (eV)")
        plt.title("Potential Energy vs Distance")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_collective_variables(self):
        """
        Plots collective variables such as total energy and radial distribution function.
        """
        import matplotlib.pyplot as plt

        # Plot total energy
        plt.figure(figsize=(10, 6))
        plt.plot(self.collective_variables["total_energy"], label="Total Energy")
        plt.xlabel("Simulation Step")
        plt.ylabel("Energy (eV)")
        plt.title("Total Energy Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot radial distribution function
        if self.collective_variables["radial_distribution"]:
            bins, rdf = self.collective_variables["radial_distribution"][-1]
            plt.figure(figsize=(10, 6))
            plt.bar(bins, rdf, width=(bins[1] - bins[0]), alpha=0.6, label="Radial Distribution Function")
            plt.xlabel("Distance (Å)")
            plt.ylabel("g(r)")
            plt.title("Radial Distribution Function")
            plt.legend()
            plt.grid(True)
            plt.show()

    def run(self):
        """
        Executes the simulation loop.
        """
        current_time = 0.0
        while current_time < self.total_time:
            self.update()
            current_time += self.dt

        self.plot_potential_energy()
        self.plot_collective_variables()
