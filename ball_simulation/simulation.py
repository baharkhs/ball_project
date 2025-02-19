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
        bond_length = 0.957
        iO = self.add_ball(mass=16.0,
                           initial_position=center_position,
                           initial_velocity=velocity,
                           species="O",
                           molecule_id=molecule_id)
        # Compute half-angle (104.5°/2) in radians.
        half_angle = np.radians(104.5) / 2
        # Use identical z offset (here 0.0 for a planar configuration)
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
        Computes forces between all pairs using Lennard-Jones potentials.
        For atoms in the same water molecule (intramolecular), uses stiff parameters to glue O and H together.
        (No angular force is applied.)
        """
        num_balls = len(self.balls)
        box_lengths = np.array([2 * self.well.radius, 2 * self.well.radius, self.well.height])
        for i in range(num_balls):
            for j in range(i + 1, num_balls):
                bi = self.balls[i]
                bj = self.balls[j]
                if bi.molecule_id is not None and bj.molecule_id is not None and (bi.molecule_id == bj.molecule_id):
                    if set([bi.species, bj.species]) == set(["O", "H"]):
                        params = {"epsilon": 6.36,
                                  "sigma": 0.957 / (2 ** (1 / 6)),
                                  "cutoff": 10.0}
                    else:
                        continue
                else:
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
                force_magnitude = 24 * params["epsilon"] * (2 * sr12 - sr6) / r
                max_force = 50.0
                force_magnitude = min(force_magnitude, max_force)
                f_lj = force_magnitude * (delta / r)
                bi.force += f_lj
                bj.force -= f_lj

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
        # Save current accelerations: a = F/m
        current_accelerations = [b.force / b.mass for b in self.balls]

        # --- Step 2: Update positions using Velocity Verlet ---
        for b in self.balls:
            # x(t + dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
            b.position += b.velocity * self.dt + 0.5 * current_accelerations[self.balls.index(b)] * self.dt ** 2
            self.well.apply_pbc(b)
            b.update_path()

        # --- Step 3: Recompute forces at the new positions ---
        for b in self.balls:
            b.force.fill(0)
        self.compute_forces()
        for b in self.balls:
            b.force += self.well.compute_wall_repulsion_force(b)
        # Save new accelerations
        new_accelerations = [b.force / b.mass for b in self.balls]

        # --- Step 4: Update velocities using the average acceleration ---
        for i, b in enumerate(self.balls):
            # v(t+dt) = v(t) + 0.5*(a(t)+a(t+dt))*dt
            b.velocity += 0.5 * (current_accelerations[i] + new_accelerations[i]) * self.dt

        # --- Step 5: Record trajectories ---
        for i, b in enumerate(self.balls):
            if i not in self.paths:
                self.paths[i] = [b.position.copy()]
            else:
                self.paths[i].append(b.position.copy())

        # --- Step 6: Apply thermostat if enabled (velocity rescaling) ---
        if rescale_temperature and (self.current_step % rescale_interval == 0):
            self.apply_velocity_rescaling(target_temperature)

        # --- Step 7: Logging: Compute & print temperature and bond distances ---
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
        """
        Creates a Simulation instance from a configuration dictionary.
        Optionally, command-line arguments (args) can override settings.
        Expected config structure:
            {
              "well": {"radius": <value>, "height": <value>},
              "simulation": {"total_time": <value>, "dt": <value>, "movement_type": "<type>", "initial_velocity": [vx,vy,vz]},
              "particles": {
                 "oxygen_molecules": [{"center_position": [...], "molecule_id": <id>}, ...],
                 "custom_particles": [ ... ]
              }
            }
        If args provides overrides (like args.vel), they will be applied.
        """
        well_radius = config["well"]["radius"]
        well_height = config["well"]["height"]
        total_time = config["simulation"]["total_time"]
        dt = config["simulation"]["dt"]
        movement_type = config["simulation"]["movement_type"]
        sim = cls(well_radius, well_height, total_time, dt, movement_type)
        sim.set_movement_type(movement_type)
        # Determine initial velocity.
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
