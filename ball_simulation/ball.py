import numpy as np
class Ball:
    def __init__(self, mass=None, initial_position=None, initial_velocity=None, species="O", molecule_id=None,
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

        # Assign charges based on species
        self.charge = -0.834 if species == "O" else 0.417  # TIP3P Model

        # Assign colors based on species, allowing overrides
        self.color = color if color else ("red" if species == "O" else "blue" if species == "H" else "gray")

        # Assign default size with overrides
        self.size = size if size else (10 if species == "O" else 6 if species == "H" else 8)

        # Assign correct mass based on species
        if mass is None:
            self.mass = 16.0 if species == "O" else 1.0  # Default mass for O and H
        else:
            self.mass = mass

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

    def compute_coulombic_force(self, other, box_lengths):
        """
        Computes the Coulombic force between two charged particles.
        """
        k_e = 1389.35  # Coulomb constant in (kJ/mol·Å·e^2), correct for distances in Å


        # Compute displacement with PBC
        delta = self.position - other.position
        delta -= box_lengths * np.round(delta / box_lengths)
        r = np.linalg.norm(delta)

        if r == 0:
            return np.zeros(3)  # Avoid division by zero

        # Coulombic force calculation
        force_magnitude = k_e * (self.charge * other.charge) / (r ** 2)
        force_direction = delta / r

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