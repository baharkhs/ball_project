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
        self.movement_type = movement_type  # "newtonian" for MD; "monte_carlo" for MC moves.
        self.current_step = 0

        self.potential_energy_data = []
        self.temperature_history = []
        self.collective_variables = {"total_energy": [], "radial_distribution": []}
        self.molecules = {}  # molecule_id -> {"O": index, "H1": index, "H2": index}
        self.paths = {}      # key: ball index, value: list of positions (trajectory)

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
        self.paths[len(self.balls) - 1] = [ball.position.copy()]
        return len(self.balls) - 1

    def create_water_molecule(self, center_position, velocity=(0, 0, 0), molecule_id=None):
        """
        Creates one water molecule (1 O and 2 H) at the given center_position.
        The O–H bond length is 0.957 Å (bond angle 104.5° is enforced via the angular potential).
        """
        bond_length = 0.957
        # Create Oxygen.
        iO = self.add_ball(mass=16.0,
                           initial_position=center_position,
                           initial_velocity=velocity,
                           species="O",
                           molecule_id=molecule_id)
        # Compute half-angle (104.5°/2) in radians.
        half_angle = np.radians(104.5) / 2
        # Use identical z offset (0.0 for a planar configuration).
        offset1 = np.array([
            bond_length * np.sin(half_angle),
            bond_length * np.cos(half_angle),
            0.0
        ])
        offset2 = np.array([
            -bond_length * np.sin(half_angle),
            bond_length * np.cos(half_angle),
            0.0
        ])
        iH1 = self.add_ball(mass=1.0,
                            initial_position=np.array(center_position) + offset1,
                            initial_velocity=velocity,
                            species="H",
                            molecule_id=molecule_id)
        iH2 = self.add_ball(mass=1.0,
                            initial_position=np.array(center_position) + offset2,
                            initial_velocity=velocity,
                            species="H",
                            molecule_id=molecule_id)
        self.molecules[molecule_id] = {"O": iO, "H1": iH1, "H2": iH2}

    def compute_forces(self):
        """
        Computes forces between all pairs. For intramolecular O-H bonds, uses a harmonic potential to maintain bond length.
        For intermolecular interactions, uses Lennard-Jones potentials.
        """
        num_balls = len(self.balls)
        box_lengths = np.array([2 * self.well.radius, 2 * self.well.radius, self.well.height])

        # Reset forces to zero before computing new forces
        for ball in self.balls:
            ball.force.fill(0)

        # Compute pairwise forces
        for i in range(num_balls):
            for j in range(i + 1, num_balls):
                bi = self.balls[i]
                bj = self.balls[j]

                # Determine if the pair is part of the same molecule (intramolecular)
                if (bi.molecule_id is not None and bj.molecule_id is not None and
                        bi.molecule_id == bj.molecule_id):
                    # Handle intramolecular O-H bonds with harmonic potential
                    if set([bi.species, bj.species]) == set(["O", "H"]):
                        k_bond = 450.0  # Spring constant in kJ/(mol·Å²)
                        r0 = 0.957  # Equilibrium bond length in Å
                        delta = bi.position - bj.position
                        # Apply periodic boundary conditions to the distance
                        delta -= box_lengths * np.round(delta / box_lengths)
                        r = np.linalg.norm(delta)
                        if r < 1e-12:  # Avoid division by zero
                            continue
                        force_magnitude = -k_bond * (r - r0)  # Harmonic force: F = -k * (r - r0)
                        f_bond = force_magnitude * (delta / r)  # Force direction along the bond
                        bi.force += f_bond
                        bj.force -= f_bond
                    # Skip H-H interactions within the same molecule (no direct bond)
                    else:
                        continue
                # Handle intermolecular interactions with Lennard-Jones potential
                else:
                    species_key = tuple(sorted([bi.species, bj.species]))
                    if species_key not in self.interaction_params:
                        continue
                    params = self.interaction_params[species_key]
                    delta = bi.position - bj.position
                    # Apply periodic boundary conditions
                    delta -= box_lengths * np.round(delta / box_lengths)
                    r = np.linalg.norm(delta)
                    if r < 1e-12 or r > params["cutoff"]:  # Avoid division by zero or beyond cutoff
                        continue
                    sr = params["sigma"] / r
                    sr6 = sr ** 6
                    sr12 = sr6 ** 2
                    force_magnitude = 24 * params["epsilon"] * (2 * sr12 - sr6) / r
                    max_force = 50.0  # Cap force for numerical stability
                    force_magnitude = min(force_magnitude, max_force)
                    f_lj = force_magnitude * (delta / r)
                    bi.force += f_lj
                    bj.force -= f_lj

    def compute_angular_forces(self, k_angle=500.0, theta_target=np.radians(104.5)):
        """
        Computes and applies angular forces for each water molecule to maintain the H–O–H angle.
        The angular potential is defined as:
            U_angle = 1/2 * k_angle * (theta - theta_target)**2,
        where theta is the current H–O–H angle.
        A simplified force distribution is applied:
          - Each hydrogen receives half the restoring force along its bond,
          - The oxygen receives the negative sum of those forces.
        """
        for molecule_id, indices in self.molecules.items():
            O = self.balls[indices["O"]]
            H1 = self.balls[indices["H1"]]
            H2 = self.balls[indices["H2"]]
            r1 = H1.position - O.position  # O->H1
            r2 = H2.position - O.position  # O->H2
            norm_r1 = np.linalg.norm(r1)
            norm_r2 = np.linalg.norm(r2)
            if norm_r1 < 1e-12 or norm_r2 < 1e-12:
                continue
            cos_theta = np.dot(r1, r2) / (norm_r1 * norm_r2)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)
            delta_theta = theta - theta_target
            # Magnitude of the angular force
            force_magnitude = -k_angle * delta_theta
            # Perpendicular components for force direction
            u1 = r1 / norm_r1
            u2 = r2 / norm_r2
            f_H1 = force_magnitude * (u2 - cos_theta * u1) / norm_r1
            f_H2 = force_magnitude * (u1 - cos_theta * u2) / norm_r2
            f_O = -(f_H1 + f_H2)
            H1.force += f_H1
            H2.force += f_H2
            O.force += f_O

    def compute_total_potential_energy(self):
        """
        Computes total potential energy (used for MC moves) by summing pairwise Lennard-Jones energies.
        """
        total_energy = 0.0
        num_balls = len(self.balls)
        box_lengths = np.array([2 * self.well.radius, 2 * self.well.radius, self.well.height])
        for i in range(num_balls):
            for j in range(i + 1, num_balls):
                bi = self.balls[i]
                bj = self.balls[j]
                species_key = tuple(sorted([bi.species, bj.species]))
                if species_key not in self.interaction_params:
                    continue
                params = self.interaction_params[species_key]
                delta = bi.position - bj.position
                delta -= box_lengths * np.round(delta / box_lengths)
                r = np.linalg.norm(delta)
                if r < 1e-8 or r > params["cutoff"]:
                    continue
                sr = params["sigma"] / r
                sr6 = sr ** 6
                sr12 = sr6 ** 2
                energy = 4 * params["epsilon"] * (sr12 - sr6)
                total_energy += energy
        return total_energy

    def perform_monte_carlo_move(self):
        """
        Performs a simple Monte Carlo move:
        Proposes random displacements and accepts the move if the total potential energy decreases.
        """
        max_disp = 0.1
        old_positions = [ball.position.copy() for ball in self.balls]
        old_energy = self.compute_total_potential_energy()
        for ball in self.balls:
            displacement = np.random.uniform(-max_disp, max_disp, size=3)
            ball.position += displacement
        for ball in self.balls:
            self.well.apply_pbc(ball)
        new_energy = self.compute_total_potential_energy()
        if new_energy > old_energy:
            for i, ball in enumerate(self.balls):
                ball.position = old_positions[i]

    def compute_system_temperature(self):
        k_B = 0.0083144621
        total_kinetic = sum(0.5 * ball.mass * np.dot(ball.velocity, ball.velocity) for ball in self.balls)
        n = len(self.balls)
        return (2 / (3 * n * k_B)) * total_kinetic if n > 0 else 0.0

    def apply_velocity_rescaling(self, target_temperature=300):
        current_temp = self.compute_system_temperature()
        if current_temp == 0:
            return
        scale = np.sqrt(target_temperature / current_temp)
        for ball in self.balls:
            ball.velocity *= scale

    def compute_potential_energy_data(self):
        """
        Computes analytical LJ potential data (for plotting) and collects simulation-based energy data.
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

        # --- Step 1: Compute forces at the current positions ---
        for b in self.balls:
            b.force.fill(0)
        self.compute_forces()
        for b in self.balls:
            b.force += self.well.compute_wall_repulsion_force(b)
        # Save current accelerations.
        current_accelerations = [b.force / b.mass for b in self.balls]

        # --- Step 2: Update positions using Velocity Verlet ---
        for i, b in enumerate(self.balls):
            b.position += b.velocity * self.dt + 0.5 * current_accelerations[i] * self.dt ** 2
            self.well.apply_pbc(b)
            b.update_path()

        # --- Step 3: Recompute forces at the new positions ---
        for b in self.balls:
            b.force.fill(0)
        self.compute_forces()
        for b in self.balls:
            b.force += self.well.compute_wall_repulsion_force(b)
        new_accelerations = [b.force / b.mass for b in self.balls]

        # --- Step 4: Update velocities using the average acceleration ---
        for i, b in enumerate(self.balls):
            b.velocity += 0.5 * (current_accelerations[i] + new_accelerations[i]) * self.dt

        # --- Step 5: Apply angular forces to maintain the H–O–H angle ---
        self.compute_angular_forces()

        # --- Step 6: Record trajectories ---
        for i, b in enumerate(self.balls):
            if i not in self.paths:
                self.paths[i] = [b.position.copy()]
            else:
                self.paths[i].append(b.position.copy())

        # --- Step 7: Thermostat: Apply velocity rescaling if enabled ---
        if rescale_temperature and (self.current_step % rescale_interval == 0):
            self.apply_velocity_rescaling(target_temperature)

        # --- Step 8: Logging: Print temperature and bond distances ---
        temperature = self.compute_system_temperature()
        for molecule_id, indices in self.molecules.items():
            O_pos = self.balls[indices["O"]].position
            H1_pos = self.balls[indices["H1"]].position
            H2_pos = self.balls[indices["H2"]].position
            d1 = np.linalg.norm(O_pos - H1_pos)
            d2 = np.linalg.norm(O_pos - H2_pos)
            print(f"Step {self.current_step}: O-H1 = {d1:.3f} Å, O-H2 = {d2:.3f} Å")
        print(f"Step {self.current_step}, Temperature: {temperature:.2f} K")
        self.temperature_history.append(temperature)
        self.compute_potential_energy_data()
        self.current_step += 1

    def run(self):
        current_time = 0.0
        while current_time < self.total_time:
            self.update()
            current_time += self.dt

    @classmethod
    def from_config(cls, config, args=None):
        well_radius = config["well"]["radius"]
        well_height = config["well"]["height"]
        total_time = config["simulation"]["total_time"]
        dt = config["simulation"]["dt"]
        movement_type = config["simulation"]["movement_type"]
        sim = cls(well_radius, well_height, total_time, dt, movement_type)
        sim.set_movement_type(movement_type)
        initial_velocity = config["simulation"].get("initial_velocity", [0.2, 0.2, 0.2])
        if args and hasattr(args, "vel") and args.vel:
            initial_velocity = [float(x) for x in args.vel.split(',')]
        for molecule in config["particles"].get("oxygen_molecules", []):
            center_position = molecule["center_position"]
            molecule_id = molecule["molecule_id"]
            sim.create_water_molecule(center_position, velocity=initial_velocity, molecule_id=molecule_id)
        for particle in config["particles"].get("custom_particles", []):
            sim.add_ball(
                mass=particle["mass"],
                initial_position=particle["position"],
                initial_velocity=particle["velocity"],
                species=particle["species"]
            )
        return sim
