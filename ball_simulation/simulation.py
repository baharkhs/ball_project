
import numpy as np
from ball_simulation.well import Well
from ball_simulation.ball import Ball

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
        self.molecules = {}

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
