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

