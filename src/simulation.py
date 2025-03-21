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
        """
        self.dt = dt
        self.total_time = total_time
        self.movement_type = movement_type
        self.interaction_params = interaction_params
        self.target_temperature = target_temperature
        self.rescale_interval = rescale_interval

        # Use the provided Well instance; if not given, create a default Well.
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

        # Now create the water molecules from the config
        for molecule in config["particles"]["oxygen_molecules"]:
            sim.create_water_molecule(
                center_position=molecule["center_position"],
                h1_z=molecule["H1_z"],
                h2_z=molecule["H2_z"],
                velocity=sim_cfg["initial_velocity"],
                molecule_id=molecule["molecule_id"]
            )

        # Create custom particles if any are defined.
        for part in config["particles"].get("custom_particles", []):
            # For a custom particle, we expect a species, a position, and a velocity.
            species = part.get("species", "X")
            position = part["position"]
            velocity = part["velocity"]
            mass = part.get("mass", ball_defaults.get("default_mass", 1.0))
            # Create a Ball for this custom particle.
            b = Ball(species, position, velocity, molecule_id=None, mass=mass)
            sim.add_ball(b)

        return sim

    def add_ball(self, ball):
        """
                Adds a Ball object to the simulation.

                - ball: a Ball object representing an individual particle.
                - Updates the paths dictionary for tracking the ball's trajectory.

                Returns the index of the added ball.
                """
        self.balls.append(ball)
        self.paths[len(self.balls) - 1] = [ball.position.copy()]
        return len(self.balls) - 1

    def create_water_molecule(self, center_position, h1_z, h2_z, velocity, molecule_id):
        """
        Method to create a water molecule. It uses the Ball class's methods (create_oxygen and create_hydrogen),
        passing the default parameters (from ball_defaults). It then groups the created Balls into a molecule,
        computes the molecule's center-of-mass (COM), and determines the fixed offsets (difference between each atom's position and the COM).

        - center_position: The initial position of the oxygen atom (and approximate center of the molecule).
        - h1_z, h2_z: The z-coordinates for the two hydrogen atoms.
        - velocity: The initial velocity vector to assign to the molecule's center-of-mass.
        - molecule_id: A unique identifier string for this molecule.
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
        """Returns list of pairs of molecule IDs for force calculations.
          This is used for calculating intermolecular forces. For each pair of molecules,
        the forces between every atom in one molecule and every atom in the other are calculated.

        """
        molecule_ids = list(self.molecules.keys())
        pairs = []
        for i in range(len(molecule_ids)):
            for j in range(i+1, len(molecule_ids)):
                pairs.append((molecule_ids[i], molecule_ids[j]))
        return pairs

    def compute_forces(self):
        """
        Computes forces on each molecule's center-of-mass (COM) by:

        1. Resetting the force on each molecule to zero.
        2. Calculating intermolecular Lennard-Jones (LJ) forces between atoms of different molecules.
           - For each pair of molecules, for every pair of atoms (one from each molecule), it calculates the distance,
             applies periodic boundary conditions, and computes the LJ force if within the cutoff.
           - The forces are then added to the corresponding molecule's net force.
        3. Calculating wall repulsion forces on each atom and adding these to the molecule's net force.
        """

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
        """    Computes the total Lennard-Jones potential energy of the system.

        It sums the pairwise potential energies for each pair of atoms (with periodic boundary conditions applied).
   """
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
        """Performs a Monte Carlo move:

        1. Saves the current positions of all balls.
        2. Displaces each ball randomly by up to max_disp in each direction.
        3. Applies periodic boundary conditions.
        4. Recomputes the total potential energy.
        5. If the new potential energy is higher than before (move is unfavorable),
           reverts to the old positions.
        """
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
        #acc_current = {mol_id: self.molecule_com[mol_id]["force"] / self.molecule_com[mol_id]["mass"]
                       #for mol_id in self.molecule_com}

        # 1) Compute forces on each ball (including the wall repulsion force).
        self.compute_forces()

        # 2) First half-step: update positions for each ball.
        for ball in self.balls:
            # Acceleration = force / mass.
            a = ball.force / ball.mass
            # Update position: new_position = old_position + velocity*dt + 0.5*a*dt^2.
            ball.position += ball.velocity * self.dt + 0.5 * a * (self.dt ** 2)

        # 3) Recompute forces after updating positions.
        self.compute_forces()

        # 4) Second half-step: update velocities for each ball.
        for ball in self.balls:
            a_new = ball.force / ball.mass
            ball.velocity += 0.5 * (a + a_new) * self.dt

            # Check if the ball is outside the cylindrical boundary in the x-y plane.
            r_xy = np.linalg.norm(ball.position[:2])
            if r_xy > self.well.radius:
                # Compute the angle in the x-y plane.
                theta = np.arctan2(ball.position[1], ball.position[0])
                # Define a 2D normal vector (only x and y components).
                normal = np.array([np.cos(theta), np.sin(theta)])
                # Reflect the x-y component of the velocity:
                v_xy = ball.velocity[:2]
                # Reflection: v' = v - 2*(v · n)*n
                ball.velocity[:2] = v_xy - 2 * np.dot(v_xy, normal) * normal
                # Place the ball at the boundary.
                ball.position[:2] = normal * self.well.radius

            # Optionally, you can also call self.well.apply_pbc(ball) for the z-axis:
            ball.position[2] %= self.well.height

        # 6) Record positions (for visualization) and temperature.
        for i, ball in enumerate(self.balls):
            self.paths[i].append(ball.position.copy())

        if rescale_temperature and (self.current_step % self.rescale_interval == 0):
            self.apply_velocity_rescaling()

        self.temperature_history.append(self.compute_system_temperature())
        self.current_step += 1

