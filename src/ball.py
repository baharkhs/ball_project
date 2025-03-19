import numpy as np

class Ball:
    def __init__(self, species, position, velocity, molecule_id=None, mass=None, color=None, size=None):
        """
        Represents a single particle (atom) in the simulation.
        """
        self.species = species
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float) if velocity is not None else np.zeros(3)
        self.molecule_id = molecule_id
        # Default mass: oxygen=16, hydrogen=1.
        if mass is None:
            self.mass = 16.0 if species == "O" else 1.0
        else:
            self.mass = mass
        # Color and size for plotting.
        self.color = color if color else ("red" if species == "O" else "blue")
        self.size = size if size else (10 if species == "O" else 6)
        # Force accumulator.
        self.force = np.zeros(3)
        # For path tracking (used in plotting).
        self.path_segments = []
        self.current_path_segment = {"x": [], "y": [], "z": []}
        self.skip_path_update = False

    @classmethod
    def create_oxygen(cls, position, velocity, molecule_id, defaults=None):
        """
                Factory method to create an oxygen atom with standard properties.

                It uses the 'defaults' dictionary (if provided) to set the mass, color, and size.

                Parameters:
                  - position: Initial [x, y, z] position.
                  - velocity: Initial velocity [vx, vy, vz].
                  - molecule_id: Identifier of the molecule.
                  - defaults: A dictionary of default parameters (e.g., "oxygen_mass", "oxygen_color", "oxygen_size").

                Returns:
                  - A new Ball instance representing an oxygen atom.
                """
        defaults = defaults or {}
        mass = defaults.get("oxygen_mass", 16.0)
        color = defaults.get("oxygen_color", "red")
        size = defaults.get("oxygen_size", 10)
        return cls("O", position, velocity, molecule_id, mass=mass, color=color, size=size)

    @classmethod
    def create_hydrogen(cls, o_position, h_z, velocity, molecule_id, sign=1, defaults=None):
        """
            create a hydrogen atom relative to an oxygen atom.

            The hydrogen's position is calculated based on a bond length and an angle.
            The 'sign' parameter indicates which hydrogen it is (one on one side, one on the opposite side).

            Parameters:
            - o_position: The oxygen's position [x, y, z]. This is the reference.
            - h_z: The desired z-coordinate for the hydrogen.
            - velocity: Initial velocity [vx, vy, vz] for the hydrogen.
            - molecule_id: Identifier for the molecule.
            - sign: Determines the direction of the x-y displacement (1 or -1).
            - defaults: A dictionary with default parameters (e.g., "bond_length", "half_angle_degrees",
                             "hydrogen_mass", "hydrogen_color", "hydrogen_size").

            Returns:
            - A new Ball instance representing a hydrogen atom.
               """
        defaults = defaults or {}
        bond_length = defaults.get("bond_length", 0.957)
        half_angle_deg = defaults.get("half_angle_degrees", 104.5)
        half_angle = np.radians(half_angle_deg) / 2
        displacement = np.array([
            sign * bond_length * np.sin(half_angle),  # x-displacement
            bond_length * np.cos(half_angle),  # y-displacement
            h_z - o_position[2]  # Difference in z from the oxygen
        ])
        h_position = np.array(o_position) + displacement
        mass = defaults.get("hydrogen_mass", 1.0)
        color = defaults.get("hydrogen_color", "blue")
        size = defaults.get("hydrogen_size", 6)
        return cls("H", h_position, velocity, molecule_id, mass=mass, color=color, size=size)

    @staticmethod
    def _compute_xy_displacement(bond_length, angle, sign=1):
        """
        Computes the x-y displacement for a hydrogen atom relative to oxygen.
        """
        return np.array([sign * bond_length * np.sin(angle), bond_length * np.cos(angle)])

    def _distance_with_pbc(self, other, box_lengths):
        """
        Calculates the distance between this ball and another, applying periodic boundary conditions.
        """
        delta = self.position - other.position
        delta -= box_lengths * np.round(delta / box_lengths)
        return np.linalg.norm(delta)

    def _direction_with_pbc(self, other, box_lengths):
        """
        Calculates the unit direction vector from this ball to another with PBC.
        Returns a zero vector if the distance is zero.
        """
        delta = self.position - other.position
        delta -= box_lengths * np.round(delta / box_lengths)
        r = np.linalg.norm(delta)
        if r == 0:
            return np.zeros(3)
        return delta / r

    def compute_interaction_force(self, other, interaction_params, box_lengths):
        """
        Computes the Lennard–Jones force between this ball and another.

        This force models the attraction and repulsion between atoms.

        Parameters:
          - other: Another Ball object.
          - interaction_params: A dictionary containing parameters "epsilon", "sigma", and "cutoff".
          - box_lengths: Dimensions of the simulation box for applying PBC.

        Returns:
          - A 3D force vector (NumPy array). If the distance is beyond the cutoff, returns a zero vector.
        """
        species_key = "-".join(sorted([self.species, other.species]))
        params = interaction_params.get(species_key, {"epsilon": 1.0, "sigma": 1.0, "cutoff": 2.5})
        delta = self.position - other.position
        delta -= box_lengths * np.round(delta / box_lengths)
        r = np.linalg.norm(delta)
        r_min = 0.8 * params["sigma"]
        if r < r_min:
            r = r_min
        if r > params["cutoff"]:
            return np.zeros(3)
        sr = params["sigma"] / r
        sr6 = sr ** 6
        sr12 = sr6 ** 2
        force_mag = 24 * params["epsilon"] * (2 * sr12 - sr6) / r
        max_force = 50.0
        force_mag = min(force_mag, max_force)
        return force_mag * (delta / r)

    def compute_coulombic_force(self, other, box_lengths):
        """
        Computes the Coulombic force between this ball and another based on their charges.
        Note: Ensure that a 'charge' attribute is set for the ball if using this function.
        """
        k_e = 1389.35  # Coulomb constant in appropriate units.
        delta = self.position - other.position
        delta -= box_lengths * np.round(delta / box_lengths)
        r = np.linalg.norm(delta)
        if r == 0:
            return np.zeros(3)
        force_magnitude = k_e * (self.charge * other.charge) / (r ** 2)
        return force_magnitude * (delta / r)

    def calculate_kinetic_energy(self):
        """
        Calculates the kinetic energy (KE) of the ball.
        """
        return 0.5 * self.mass * np.dot(self.velocity, self.velocity)

    def calculate_temperature(self):
        """
        Calculates the temperature based on the ball's kinetic energy.
        Assumes a single-particle model (for reference).
        """
        k_B = 0.0083144621  # Boltzmann constant in appropriate units.
        kinetic_energy = self.calculate_kinetic_energy()
        return (2 / 3) * (kinetic_energy / k_B)

    def compute_intra_molecular_forces(self, other, bond_length=1.0, angle=104.5, k_bond=100, k_angle=50):
        """
        Computes harmonic forces to maintain bonds and angles within a molecule.
        Returns the bond force as per Hooke's law.
        (Angle forces are not fully implemented here.)
        """
        delta = self.position - other.position
        r = np.linalg.norm(delta)
        bond_force = -k_bond * (r - bond_length) * (delta / r)
        return bond_force

    def update_path(self):
        """
        Updates the ball's path for visualization.
        Handles PBC transitions and starts a new segment if needed.
        """
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
        """
        Finalizes the current path segment at the end of the simulation.
        """
        if self.current_path_segment["x"]:
            self.path_segments.append(self.current_path_segment)

    def get_path_segments(self):
        """
        Returns all recorded path segments.
        """
        return self.path_segments + [self.current_path_segment]

    def finalize_simulation(self):
        """
        Finalizes path tracking at the end of the simulation.
        """
        self.finalize_path()

    @staticmethod
    def lennard_jones_potential(r, epsilon, sigma):
        """
        Computes the Lennard–Jones potential energy for a given distance.
        Returns infinity if r is zero.
        """
        if r == 0:
            return np.inf
        sr = sigma / r
        sr6 = sr ** 6
        sr12 = sr6 ** 2
        return 4 * epsilon * (sr12 - sr6)

    @staticmethod
    def lennard_jones_force(r, epsilon, sigma):
        """
        Computes the Lennard–Jones force magnitude for a given distance.
        Returns infinity if r is zero.
        """
        if r == 0:
            return np.inf
        sr = sigma / r
        sr6 = sr ** 6
        sr12 = sr6 ** 2
        return 24 * epsilon * (2 * sr12 - sr6) / r
