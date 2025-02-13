import numpy as np
from ball_simulation.well import Well
from ball_simulation.ball import Ball


class Simulation:
    def __init__(self, well_radius=0.6, well_height=2.0, total_time=15.0, dt=0.001, movement_type="newtonian"):
        """
        Initializes the simulation.
        """
        self.well = Well(well_radius, well_height)
        self.dt = dt
        self.total_time = total_time
        self.balls = []
        self.movement_type = movement_type
        self.current_step = 0

        self.potential_energy_data = []
        self.temperature_history = []

        # Dictionary: molecule_id -> {"O": index, "H1": index, "H2": index}
        self.molecules = {}

        # Standard nonbonded Lennard-Jones parameters (for atoms in different molecules)
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
            initial_position=np.array(initial_position, dtype=float),
            initial_velocity=np.array(initial_velocity, dtype=float) if initial_velocity is not None else np.zeros(3),
            species=species,
            molecule_id=molecule_id,
            color=color,
            size=size
        )
        self.balls.append(ball)
        return len(self.balls) - 1

    def create_water_molecule(self, center_position, velocity=(0, 0, 0), molecule_id=None):
        """
        Creates one water molecule (1 O and 2 H) at the given center_position.
        The O–H bond length is 0.957 Å.
        (We ignore the bond angle for simplicity.)
        """
        bond_length = 0.957
        # For simplicity, we use fixed offsets.
        # You can adjust the offsets if needed.
        iO = self.add_ball(
            mass=16.0,
            initial_position=center_position,
            initial_velocity=velocity,
            species="O",
            molecule_id=molecule_id
        )
        # Place two hydrogens with a slight offset in z so they don't completely overlap.
        offset1 = np.array([bond_length, 0.0, 0.1])
        offset2 = np.array([-bond_length, 0.0, -0.1])
        iH1 = self.add_ball(
            mass=1.0,
            initial_position=np.array(center_position) + offset1,
            initial_velocity=velocity,
            species="H",
            molecule_id=molecule_id
        )
        iH2 = self.add_ball(
            mass=1.0,
            initial_position=np.array(center_position) + offset2,
            initial_velocity=velocity,
            species="H",
            molecule_id=molecule_id
        )
        self.molecules[molecule_id] = {"O": iO, "H1": iH1, "H2": iH2}

    def compute_forces(self):
        """
        Computes forces between all pairs using Lennard-Jones potentials.
        For atoms in the same water molecule (i.e. intramolecular), we use a different set
        of LJ parameters to “glue” O and H together.
        No angular force is applied.
        """
        num_balls = len(self.balls)
        # Periodic boundary box dimensions.
        box_lengths = np.array([2 * self.well.radius, 2 * self.well.radius, self.well.height])

        for i in range(num_balls):
            for j in range(i + 1, num_balls):
                bi = self.balls[i]
                bj = self.balls[j]

                # If both belong to the same molecule, use intramolecular parameters
                if (bi.molecule_id is not None and bj.molecule_id is not None and
                        bi.molecule_id == bj.molecule_id):
                    if set([bi.species, bj.species]) == set(["O", "H"]):
                        # Intramolecular O–H: parameters chosen so that the LJ minimum is at ~0.957 Å.
                        params = {"epsilon": 6.36,
                                  "sigma": 0.957 / (2 ** (1 / 6)),  # ~0.8527 Å
                                  "cutoff": 10.0}  # Use a long cutoff so the force is always active
                    else:
                        # For intramolecular pairs that aren't O–H (like H–H), skip force calculation.
                        continue
                else:
                    # Use standard intermolecular parameters for atoms in different molecules.
                    species_key = tuple(sorted([bi.species, bj.species]))
                    if species_key not in self.interaction_params:
                        continue
                    params = self.interaction_params[species_key]

                # Calculate the displacement vector, accounting for periodic boundaries.
                delta = bi.position - bj.position
                delta -= box_lengths * np.round(delta / box_lengths)
                r = np.linalg.norm(delta)
                if r < 1e-8 or r > params["cutoff"]:
                    continue
                sr = params["sigma"] / r
                sr6 = sr ** 6
                sr12 = sr6 ** 2
                # LJ force magnitude: derivative of the potential.
                force_magnitude = 24 * params["epsilon"] * (2 * sr12 - sr6) / r
                max_force = 50.0
                force_magnitude = min(force_magnitude, max_force)
                f_lj = force_magnitude * (delta / r)
                # Newton's third law: apply equal and opposite forces.
                bi.force += f_lj
                bj.force -= f_lj

    def compute_system_temperature(self):
        k_B = 0.0083144621  # Boltzmann constant in (kJ/mol)/K
        total_kinetic_energy = sum(0.5 * ball.mass * np.dot(ball.velocity, ball.velocity)
                                   for ball in self.balls)
        n = len(self.balls)
        return (2 / (3 * n * k_B)) * total_kinetic_energy if n > 0 else 0.0

    def apply_velocity_rescaling(self, target_temperature=300):
        current_temperature = self.compute_system_temperature()
        if current_temperature == 0:
            return
        scaling_factor = np.sqrt(target_temperature / current_temperature)
        for ball in self.balls:
            ball.velocity *= scaling_factor

    def compute_potential_energy_data(self):
        """
        Computes analytical Lennard-Jones potential data (for plotting)
        and collects a simulation-based energy data point (for O–O interactions).
        """
        pair_key = ("O", "O")
        epsilon, sigma = self.interaction_params[pair_key]["epsilon"], self.interaction_params[pair_key]["sigma"]
        r_analytical = np.linspace(0.1, 3.0, 200)
        lj_analytical = 4 * epsilon * ((sigma / r_analytical) ** 12 - (sigma / r_analytical) ** 6)
        self.analytical_potential_energy_data = list(zip(r_analytical, lj_analytical))

        oxygens = [b for b in self.balls if b.species == "O"]
        if len(oxygens) != 2:
            return [], self.analytical_potential_energy_data
        box_lengths = np.array([2 * self.well.radius, 2 * self.well.radius, self.well.height])
        delta = oxygens[0].position - oxygens[1].position
        delta -= box_lengths * np.round(delta / box_lengths)
        r_simulated = np.linalg.norm(delta)
        lj_simulated = 4 * epsilon * ((sigma / r_simulated) ** 12 - (sigma / r_simulated) ** 6)
        self.potential_energy_data.append((r_simulated, lj_simulated))
        return self.potential_energy_data, self.analytical_potential_energy_data

    def update(self, rescale_temperature=True, target_temperature=300, rescale_interval=50):
        if not self.balls:
            print("No balls in the simulation.")
            return

        # Reset forces on all balls
        for b in self.balls:
            b.force.fill(0)

        # Compute forces (both intra- and intermolecular)
        self.compute_forces()

        # Add wall repulsion forces
        for b in self.balls:
            b.force += self.well.compute_wall_repulsion_force(b)

        # Update positions and velocities
        for b in self.balls:
            acceleration = b.force / b.mass
            b.velocity += acceleration * self.dt
            b.position += b.velocity * self.dt
            self.well.apply_pbc(b)
            b.update_path()

        # Optionally apply a simple thermostat
        if rescale_temperature and (self.current_step % rescale_interval == 0):
            self.apply_velocity_rescaling(target_temperature)

        # Log temperature
        temperature = self.compute_system_temperature()
        print(f"Step {self.current_step}, Temperature: {temperature:.2f} K")
        self.temperature_history.append(temperature)

        # Update potential energy data
        self.compute_potential_energy_data()

        self.current_step += 1

    def run(self):
        current_time = 0.0
        while current_time < self.total_time:
            self.update()
            current_time += self.dt
