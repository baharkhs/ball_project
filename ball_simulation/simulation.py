import numpy as np
from ball_simulation.well import Well
from ball_simulation.ball import Ball

class Simulation:
    def __init__(self, well_radius=0.5, well_height=1.0, total_time=10.0, dt=0.001, movement_type="newtonian"):
        """
        Initializes the simulation environment, parameters, and balls.
        """
        self.well = Well(well_radius, well_height)  # Cylindrical well
        self.dt = dt
        self.total_time = total_time
        self.balls = []
        self.movement_type = movement_type
        self.current_step = 0

        self.potential_energy_data = []
        self.temperature_history = []
        self.collective_variables = {
            "total_energy": [],
            "radial_distribution": []
        }

        # Track molecules
        self.molecules = {}

        # Lennard-Jones interaction parameters
        self.interaction_params = {
            ("H", "H"): {"epsilon": 0.3, "sigma": 0.1, "cutoff": 2.5},
            ("O", "O"): {"epsilon": 0.8, "sigma": 0.3, "cutoff": 5.0},
            ("H", "O"): {"epsilon": 0.5, "sigma": 0.2, "cutoff": 4.0}
        }

    def set_movement_type(self, movement_type="newtonian"):
        self.movement_type = movement_type

    def add_ball(self, mass=1.0, initial_position=None, initial_velocity=None,
                 species="O", molecule_id=None, color=None, size=None):
        if initial_position is None:
            initial_position = [0.0, 0.0, 0.0]

        ball = Ball(
            mass=mass,
            initial_position=initial_position,
            initial_velocity=initial_velocity,
            species=species,
            molecule_id=molecule_id,
            color=color,
            size=size
        )
        self.balls.append(ball)
        return len(self.balls) - 1  # return index

    def create_water_molecule(self, center_position, velocity=(0, 0, 0), molecule_id=None):
        """
        Creates an H2O molecule (1 O + 2 H) and records their indices in self.molecules.
        """
        bond_length = 0.957
        angle_deg = 104.5
        angle_rad = np.radians(angle_deg)

        # 1) Add Oxygen
        iO = self.add_ball(
            mass=16.0,
            initial_position=np.array(center_position),
            initial_velocity=np.array(velocity),
            species="O",
            molecule_id=molecule_id
        )

        # 2) Compute offsets for the two H
        offset1 = np.array([
            bond_length * np.sin(angle_rad / 2),
            bond_length * np.cos(angle_rad / 2),
            0.1
        ])
        offset2 = np.array([
            -bond_length * np.sin(angle_rad / 2),
            bond_length * np.cos(angle_rad / 2),
            -0.1
        ])

        iH1 = self.add_ball(
            mass=1.0,
            initial_position=np.array(center_position) + offset1,
            initial_velocity=np.array(velocity),
            species="H",
            molecule_id=molecule_id
        )
        iH2 = self.add_ball(
            mass=1.0,
            initial_position=np.array(center_position) + offset2,
            initial_velocity=np.array(velocity),
            species="H",
            molecule_id=molecule_id
        )

        self.molecules[molecule_id] = {"O": iO, "H1": iH1, "H2": iH2}

    def compute_intermolecular_forces(self):
        num_balls = len(self.balls)
        box_lengths = np.array([2 * self.well.radius, 2 * self.well.radius, self.well.height])

        for i in range(num_balls):
            for j in range(i + 1, num_balls):
                bi, bj = self.balls[i], self.balls[j]
                if bi.molecule_id == bj.molecule_id:
                    continue  # Ignore intra-molecular forces

                f_ij = bi.compute_interaction_force(bj, self.interaction_params, box_lengths)
                bi.force += f_ij
                bj.force -= f_ij

    def compute_system_temperature(self):
        k_B = 0.0083144621
        total_kinetic_energy = sum(
            0.5 * ball.mass * np.dot(ball.velocity, ball.velocity)
            for ball in self.balls
        )
        n = len(self.balls)
        if n == 0:
            return 0.0
        return (2 / (3 * n * k_B)) * total_kinetic_energy

    def apply_velocity_rescaling(self, target_temperature=300):
        current_temperature = self.compute_system_temperature()
        scaling_factor = np.sqrt(target_temperature / current_temperature)
        for ball in self.balls:
            ball.velocity *= scaling_factor

    def compute_potential_energy_data(self):
        """
        Calculate potential energy only between the two oxygen atoms.
        """
        oxygens = [b for b in self.balls if b.species == "O"]
        if len(oxygens) != 2:
            return [], []

        box_lengths = np.array([2 * self.well.radius, 2 * self.well.radius, self.well.height])

        delta = oxygens[0].position - oxygens[1].position
        delta -= box_lengths * np.round(delta / box_lengths)
        r = np.linalg.norm(delta)

        pair_key = ("O", "O")
        params = self.interaction_params[pair_key]
        epsilon, sigma = params["epsilon"], params["sigma"]
        potential_energy = Ball.lennard_jones_potential(r, epsilon, sigma)

        self.potential_energy_data.append((r, potential_energy))
        return self.potential_energy_data

    def update(self, rescale_temperature=True, target_temperature=300, rescale_interval=50):
        if not self.balls:
            print("No balls in the simulation.")
            return

        # Reset forces
        for b in self.balls:
            b.force.fill(0)

        # Compute intermolecular forces
        self.compute_intermolecular_forces()

        # Apply wall repulsion
        for b in self.balls:
            b.force += self.well.compute_wall_repulsion_force(b)

        # Update positions and velocities
        for b in self.balls:
            acceleration = b.force / b.mass
            b.velocity += acceleration * self.dt
            b.position += b.velocity * self.dt
            self.well.apply_pbc(b)
            b.update_path()

        # Apply thermostat
        if rescale_temperature and (self.current_step % rescale_interval == 0):
            self.apply_velocity_rescaling(target_temperature)

        # Track temperature
        temperature = self.compute_system_temperature()
        print(f"Step {self.current_step}, Temperature: {temperature:.2f} K")
        self.temperature_history.append(temperature)

        self.current_step += 1

        # Compute potential energy data for oxygen atoms
        self.compute_potential_energy_data()

    def run(self):
        current_time = 0.0
        while current_time < self.total_time:
            self.update()
            current_time += self.dt
