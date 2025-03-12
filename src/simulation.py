import numpy as np
from src.ball import Ball
from src.well import Well

class Simulation:
    def __init__(
        self,
        dt,
        total_time,
        movement_type,
        interaction_params,
        target_temperature,
        rescale_interval,
        well_instance=None,  # <--- Must accept well_instance
        ball_defaults=None   # <--- Must accept ball_defaults
    ):
        """
        Creates a Simulation with the specified parameters and a Well instance.
        If well_instance is None, you can optionally create a default well.
        """
        self.dt = dt
        self.total_time = total_time
        self.movement_type = movement_type
        self.interaction_params = interaction_params
        self.target_temperature = target_temperature
        self.rescale_interval = rescale_interval

        # If we passed in a custom well, use it; else create a default one
        if well_instance is not None:
            self.well = well_instance
        else:
            self.well = Well(5.0, 4.0)  # fallback if you want

        # Store ball defaults from config (masses, colors, bond lengths, etc.)
        if ball_defaults is None:
            ball_defaults = {}
        self.ball_defaults = ball_defaults

        # Containers for simulation state
        self.balls = []
        self.molecules = {}
        self.molecule_com = {}
        self.paths = {}
        self.temperature_history = []
        self.potential_energy_data = []
        self.current_step = 0

    @classmethod
    def from_config(cls, config):
        """
        Reads parameters from a config dict and returns a fully set-up Simulation.
        """
        sim_cfg = config["simulation"]
        well_cfg = config["well"]

        # Build a Well from the config
        well = Well(
            radius=well_cfg["radius"],
            height=well_cfg["height"],
            wall_decay_length=well_cfg.get("wall_decay_length", 0.05),
            repulsion_constant=well_cfg.get("repulsion_constant", 500.0),
            atom_radius=well_cfg.get("atom_radius", 0.1)
        )

        # Interaction parameters from config
        interaction_params = config.get("interaction_params", {})

        # Ball defaults (mass, color, bond_length, etc.)
        ball_defaults = config.get("ball", {})

        # Create the Simulation instance
        sim = cls(
            dt=sim_cfg["dt"],
            total_time=sim_cfg["total_time"],
            movement_type=sim_cfg.get("movement_type", "newtonian"),
            interaction_params=interaction_params,
            target_temperature=sim_cfg["target_temperature"],
            rescale_interval=sim_cfg["rescale_interval"],
            well_instance=well,         # Pass the Well object
            ball_defaults=ball_defaults # Pass the ball defaults
        )

        # Now create the water molecules (or other particles) from the config
        for molecule in config["particles"]["oxygen_molecules"]:
            sim.create_water_molecule(
                center_position=molecule["center_position"],
                h1_z=molecule["H1_z"],
                h2_z=molecule["H2_z"],
                velocity=sim_cfg["initial_velocity"],
                molecule_id=molecule["molecule_id"]
            )

        return sim

    def add_ball(self, ball):
        self.balls.append(ball)
        self.paths[len(self.balls) - 1] = [ball.position.copy()]
        return len(self.balls) - 1

    def create_water_molecule(self, center_position, h1_z, h2_z, velocity, molecule_id):
        """
        Example method to create a water molecule.
        Adjust if your Ball class uses different creation methods.
        """
        from .ball import Ball  # or you already have it imported at top

        center_position = np.array(center_position, dtype=float)
        velocity = np.array(velocity, dtype=float)

        # Example usage of Ball's class methods that take 'defaults' as a param
        oxygen = Ball.create_oxygen(center_position, velocity, molecule_id, defaults=self.ball_defaults)
        h1 = Ball.create_hydrogen(center_position, h1_z, velocity, molecule_id, sign=1, defaults=self.ball_defaults)
        h2 = Ball.create_hydrogen(center_position, h2_z, velocity, molecule_id, sign=-1, defaults=self.ball_defaults)

        iO = self.add_ball(oxygen)
        iH1 = self.add_ball(h1)
        iH2 = self.add_ball(h2)

        self.molecules[molecule_id] = {"O": iO, "H1": iH1, "H2": iH2}

        positions = [self.balls[iO].position, self.balls[iH1].position, self.balls[iH2].position]
        masses = [self.balls[iO].mass, self.balls[iH1].mass, self.balls[iH2].mass]
        com_position = np.average(positions, weights=masses, axis=0)
        self.molecule_com[molecule_id] = {
            "position": com_position,
            "velocity": velocity.copy(),
            "force": np.zeros(3),
            "mass": sum(masses),
            "offsets": {
                "O": self.balls[iO].position - com_position,
                "H1": self.balls[iH1].position - com_position,
                "H2": self.balls[iH2].position - com_position
            }
        }

    def _get_molecule_pairs(self):
        """Returns list of pairs of molecule IDs for force calculations."""
        molecule_ids = list(self.molecules.keys())
        pairs = []
        for i in range(len(molecule_ids)):
            for j in range(i+1, len(molecule_ids)):
                pairs.append((molecule_ids[i], molecule_ids[j]))
        return pairs

    def compute_forces(self):
        """Computes forces on each molecule's COM from Lennard-Jones and wall repulsion."""
        # Reset forces
        for mol_id in self.molecule_com:
            self.molecule_com[mol_id]["force"].fill(0)
        box_lengths = np.array([2 * self.well.radius, 2 * self.well.radius, self.well.height])
        # Intermolecular Lennard-Jones forces.
        for mol1_id, mol2_id in self._get_molecule_pairs():
            for idx1 in self.molecules[mol1_id].values():
                for idx2 in self.molecules[mol2_id].values():
                    b1 = self.balls[idx1]
                    b2 = self.balls[idx2]
                    species_key = "-".join(sorted([b1.species, b2.species]))
                    if species_key not in self.interaction_params:
                        continue
                    params = self.interaction_params[species_key]
                    delta = b1.position - b2.position
                    delta -= box_lengths * np.round(delta / box_lengths)
                    r = np.linalg.norm(delta)
                    if r == 0 or r > params["cutoff"]:
                        continue
                    sr = params["sigma"] / r
                    sr6 = sr ** 6
                    sr12 = sr6 ** 2
                    force_mag = 24 * params["epsilon"] * (2 * sr12 - sr6) / r
                    max_force = 50.0
                    force_mag = min(force_mag, max_force)
                    f_vec = force_mag * (delta / r)
                    self.molecule_com[mol1_id]["force"] += f_vec
                    self.molecule_com[mol2_id]["force"] -= f_vec

        # Wall repulsion for each atom.
        for mol_id, indices in self.molecules.items():
            for atom_idx in indices.values():
                ball = self.balls[atom_idx]
                wall_force = self.well.compute_wall_repulsion_force(ball)
                self.molecule_com[mol_id]["force"] += wall_force

    def update_molecule_positions(self, molecule_id):
        """Updates positions of all atoms in a molecule based on its COM and stored offsets."""
        com_pos = self.molecule_com[molecule_id]["position"]
        offsets = self.molecule_com[molecule_id]["offsets"]
        for label, idx in self.molecules[molecule_id].items():
            self.balls[idx].position = com_pos + offsets[label]
            self.well.apply_pbc(self.balls[idx])

    def compute_total_potential_energy(self):
        """Computes total Lennard-Jones potential energy of the system."""
        total_energy = 0.0
        num_balls = len(self.balls)
        box_lengths = np.array([2 * self.well.radius, 2 * self.well.radius, self.well.height])
        for i in range(num_balls):
            for j in range(i+1, num_balls):
                b1 = self.balls[i]
                b2 = self.balls[j]
                species_key = "-".join(sorted([b1.species, b2.species]))
                if species_key not in self.interaction_params:
                    continue
                params = self.interaction_params[species_key]
                delta = b1.position - b2.position
                delta -= box_lengths * np.round(delta / box_lengths)
                r = np.linalg.norm(delta)
                if r == 0 or r > params["cutoff"]:
                    continue
                sr = params["sigma"] / r
                sr6 = sr ** 6
                sr12 = sr6 ** 2
                energy = 4 * params["epsilon"] * (sr12 - sr6)
                total_energy += energy
        return total_energy

    def compute_system_temperature(self):
        """Calculates system temperature from COM kinetic energies."""
        k_B = 0.0083144621
        total_kinetic = 0.0
        for mol in self.molecule_com.values():
            total_kinetic += 0.5 * mol["mass"] * np.dot(mol["velocity"], mol["velocity"])
        n_mol = len(self.molecule_com)
        return (2 / (3 * n_mol * k_B)) * total_kinetic if n_mol > 0 else 0.0

    def apply_velocity_rescaling(self):
        """Rescales molecule velocities to maintain target temperature."""
        current_temp = self.compute_system_temperature()
        if current_temp == 0:
            return
        scale = np.sqrt(self.target_temperature / current_temp)
        for mol in self.molecule_com.values():
            mol["velocity"] *= scale

    def perform_monte_carlo_move(self, max_disp=0.1):
        """Proposes a Monte Carlo move and accepts if total potential energy decreases."""
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

    def update(self, rescale_temperature=True):
        """
        Advances the simulation by one time step using a Velocity Verlet integrator.
        1. Compute forces (which may be zero if molecules are far apart, so initial velocity is key).
        2. Update positions based on current velocity and acceleration.
        3. Recompute forces.
        4. Update velocities.
        5. Apply periodic boundary conditions and record paths/temperature.
        """
        # Compute forces and accelerations at current time.
        self.compute_forces()
        acc_current = {mol_id: self.molecule_com[mol_id]["force"] / self.molecule_com[mol_id]["mass"]
                       for mol_id in self.molecule_com}

        # First half-step: update center-of-mass positions of each molecule.
        for mol_id, com in self.molecule_com.items():
            new_pos = com["position"] + com["velocity"] * self.dt + 0.5 * acc_current[mol_id] * (self.dt ** 2)
            # Enforce periodic boundary conditions for z.
            if new_pos[2] > self.well.height:
                new_pos[2] -= self.well.height
            elif new_pos[2] < 0:
                new_pos[2] += self.well.height
            # For x-y, if outside the cylinder, project back inside.
            r_xy = np.linalg.norm(new_pos[:2])
            if r_xy > self.well.radius:
                theta = np.arctan2(new_pos[1], new_pos[0])
                new_pos[0] = self.well.radius * np.cos(theta)
                new_pos[1] = self.well.radius * np.sin(theta)
            com["position"] = new_pos
            self.update_molecule_positions(mol_id)

        # Recompute forces after updating positions.
        self.compute_forces()
        acc_new = {mol_id: self.molecule_com[mol_id]["force"] / self.molecule_com[mol_id]["mass"]
                   for mol_id in self.molecule_com}

        # Second half-step: update velocities.
        for mol_id in self.molecule_com:
            self.molecule_com[mol_id]["velocity"] += 0.5 * (acc_current[mol_id] + acc_new[mol_id]) * self.dt

        # Record positions (for visualization) and temperature.
        for i, ball in enumerate(self.balls):
            self.paths[i].append(ball.position.copy())

        if rescale_temperature and (self.current_step % self.rescale_interval == 0):
            self.apply_velocity_rescaling()

        self.temperature_history.append(self.compute_system_temperature())
        self.current_step += 1

