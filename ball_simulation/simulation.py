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

        # Track which atoms form each water molecule: {molecule_id: {"O": iO, "H1": iH1, "H2": iH2}}
        self.molecules = {}

        # LJ interaction params
        self.interaction_params = {
            ("H", "H"): {"epsilon": 1.0, "sigma": 0.1, "cutoff": 2.5},
            ("O", "O"): {"epsilon": 2.0, "sigma": 0.2, "cutoff": 5.0},
            ("H", "O"): {"epsilon": 1.5, "sigma": 0.15, "cutoff": 4.0}
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

        if molecule_id not in self.molecules:
            self.molecules[molecule_id] = {}
        self.molecules[molecule_id]["O"]  = iO
        self.molecules[molecule_id]["H1"] = iH1
        self.molecules[molecule_id]["H2"] = iH2

    # -------------------------------------------------------------------------
    #          Intermolecular vs Intramolecular Forces

    def compute_intermolecular_forces(self):
        """
        Lennard-Jones *only* for pairs in *different* molecules.
        """
        num_balls = len(self.balls)
        box_lengths = np.array([2*self.well.radius, 2*self.well.radius, self.well.height])

        for i in range(num_balls):
            for j in range(i+1, num_balls):
                bi, bj = self.balls[i], self.balls[j]
                # skip if same molecule
                if (bi.molecule_id is not None and
                    bj.molecule_id is not None and
                    bi.molecule_id == bj.molecule_id):
                    continue

                f_ij = bi.compute_interaction_force(bj, self.interaction_params, box_lengths)
                bi.force += f_ij
                bj.force -= f_ij

    def compute_intramolecular_forces(self):
        """
        Apply harmonic bond & angle to each H2O molecule so it stays together.
        Lower constants to avoid huge velocities.
        """
        k_bond  = 30.0  # was 100.0 in your code
        r0      = 0.957
        k_angle = 15.0  # was 50.0
        angle0  = np.radians(104.5)

        for m_id, atoms in self.molecules.items():
            iO  = atoms["O"]
            iH1 = atoms["H1"]
            iH2 = atoms["H2"]

            bO  = self.balls[iO]
            bH1 = self.balls[iH1]
            bH2 = self.balls[iH2]

            # Bond O-H1
            F_OH1 = self.compute_bond_force(bO, bH1, r0, k_bond)
            bO.force  -= F_OH1
            bH1.force += F_OH1

            # Bond O-H2
            F_OH2 = self.compute_bond_force(bO, bH2, r0, k_bond)
            bO.force  -= F_OH2
            bH2.force += F_OH2

            # Angle H1-O-H2
            fO, fH1, fH2 = self.compute_angle_force(bO, bH1, bH2, angle0, k_angle)
            bO.force  += fO
            bH1.force += fH1
            bH2.force += fH2

    @staticmethod
    def compute_bond_force(b1, b2, r0, k_bond):
        """
        Harmonic bond: F = -k (r - r0) (delta / r).
        """
        delta = b1.position - b2.position
        r = np.linalg.norm(delta)
        if r < 1e-8:
            return np.zeros(3)
        vec = (r - r0)*(delta/r)
        return -k_bond*vec

    @staticmethod
    def compute_angle_force(bO, bH1, bH2, angle0, k_angle):
        """
        H1-O-H2 harmonic angle. Return forces on (O, H1, H2).
        """
        RO  = bO.position
        RH1 = bH1.position
        RH2 = bH2.position

        u = RH1 - RO
        v = RH2 - RO
        ru = np.linalg.norm(u)
        rv = np.linalg.norm(v)
        if ru<1e-8 or rv<1e-8:
            return np.zeros(3), np.zeros(3), np.zeros(3)

        cos_t = np.dot(u, v)/(ru*rv)
        cos_t = max(-1, min(1, cos_t))
        theta = np.arccos(cos_t)
        sin_t = np.sqrt(1 - cos_t*cos_t)
        if sin_t<1e-8:
            return np.zeros(3), np.zeros(3), np.zeros(3)

        dE_dtheta = k_angle*(theta - angle0)
        factor = dE_dtheta/sin_t

        fH1 = factor*((v/(ru*rv)) - (cos_t*u/(ru*ru)))
        fH2 = factor*((u/(ru*rv)) - (cos_t*v/(rv*rv)))
        fO  = -(fH1 + fH2)
        return fO, fH1, fH2


    # -------------------------------------------------------------------------

    def apply_monte_carlo_perturbation(self, ball, k_B=0.0083144621, temperature=300):
        old_pos = ball.position.copy()
        old_en = self.calculate_particle_energy(ball)
        pert = np.random.uniform(-0.1, 0.1, size=3)
        ball.position += pert

        new_en = self.calculate_particle_energy(ball)
        delta_e = new_en - old_en
        if delta_e>0 and np.random.rand()>np.exp(-delta_e/(k_B*temperature)):
            ball.position = old_pos

    def calculate_particle_energy(self, ball):
        """
        Sums squares of LJ + wall forces for that ball, as you had.
        """
        energy = 0.0
        # wall repulsion
        energy += np.sum(self.well.compute_wall_repulsion_force(ball)**2)

        # pairwise
        for other in self.balls:
            if other is not ball:
                f_ij = ball.compute_interaction_force(
                    other,
                    self.interaction_params,
                    np.array([2*self.well.radius, 2*self.well.radius, self.well.height])
                )
                energy += np.sum(f_ij**2)
        return energy

    def compute_potential_energy_data(self):
        """
        Potential energy vs distance for all pairs, using LJ (like your code).
        """
        distances = []
        potential_energies = []
        n = len(self.balls)
        box_lengths = np.array([2*self.well.radius, 2*self.well.radius, self.well.height])

        for i in range(n):
            for j in range(i+1, n):
                b1, b2 = self.balls[i], self.balls[j]
                delta = b1.position - b2.position
                delta -= box_lengths*np.round(delta/box_lengths)
                r = np.linalg.norm(delta)

                pair_key = tuple(sorted([b1.species, b2.species]))
                params = self.interaction_params.get(pair_key, {"epsilon":1.0,"sigma":1.0})
                epsilon, sigma = params["epsilon"], params["sigma"]
                pe = Ball.lennard_jones_potential(r, epsilon, sigma)
                distances.append(r)
                potential_energies.append(pe)

        return distances, potential_energies

    def compute_radial_distribution_function(self, bins=50):
        n = len(self.balls)
        box_lengths = np.array([2*self.well.radius, 2*self.well.radius, self.well.height])
        distances = []
        for i in range(n):
            for j in range(i+1, n):
                delta = self.balls[i].position - self.balls[j].position
                delta -= box_lengths*np.round(delta/box_lengths)
                distances.append(np.linalg.norm(delta))

        hist, bin_edges = np.histogram(distances, bins=bins, density=True)
        self.collective_variables["radial_distribution"].append((bin_edges[:-1], hist))

    def update(self, rescale_temperature=True, target_temperature=300, rescale_interval=100):
        if not self.balls:
            print("No balls in the simulation.")
            return

        box_lengths = np.array([2*self.well.radius, 2*self.well.radius, self.well.height])

        # 1) zero forces
        for b in self.balls:
            b.force.fill(0)

        # 2) Intermolecular LJ
        self.compute_intermolecular_forces()

        # 3) Intramolecular O–H bond & angle
        self.compute_intramolecular_forces()

        # 4) Wall repulsion
        for b in self.balls:
            b.force += self.well.compute_wall_repulsion_force(b)

        # 5) MC or Newtonian
        if self.movement_type=="monte_carlo":
            for b in self.balls:
                self.apply_monte_carlo_perturbation(b)
        else:
            for b in self.balls:
                accel = b.force/b.mass
                b.velocity += accel*self.dt
                b.position += b.velocity*self.dt
                self.well.apply_pbc(b)
                b.update_path()

        # 6) Thermostat if needed
        if rescale_temperature and (self.current_step % rescale_interval==0):
            self.apply_velocity_rescaling(target_temperature)

        temp = self.compute_system_temperature()
        print(f"Step {self.current_step}, Temperature: {temp:.2f} K")
        self.temperature_history.append(temp)

        self.current_step += 1

        # total energy
        total_energy = sum(self.calculate_particle_energy(b) for b in self.balls)
        self.collective_variables["total_energy"].append(total_energy)

        # RDF
        self.compute_radial_distribution_function()

    def apply_velocity_rescaling(self, target_temperature):
        k_B = 0.0083144621
        total_ke = 0.0
        for b in self.balls:
            v2 = np.dot(b.velocity, b.velocity)
            total_ke += 0.5*b.mass*v2
        n = len(self.balls)
        if n==0 or total_ke<1e-10:
            return

        curr_temp = (2/(3*n*k_B))*total_ke
        scale = np.sqrt(target_temperature/curr_temp)
        for b in self.balls:
            b.velocity *= scale

    def compute_system_temperature(self):
        k_B = 0.0083144621
        total_ke = sum(
            0.5*b.mass*np.dot(b.velocity, b.velocity)
            for b in self.balls
        )
        n = len(self.balls)
        if n==0:
            return 0.0
        return (2/(3*n*k_B))*total_ke

    def run(self):
        current_time = 0.0
        while current_time<self.total_time:
            self.update()
            current_time += self.dt

        self.plot_potential_energy()
        self.plot_collective_variables()


