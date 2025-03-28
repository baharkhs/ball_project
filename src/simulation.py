# simulation.py
import numpy as np
from src.ball import Ball
from src.well import Well
import warnings
import random

# --- Conversion Factors ---
# Internal Energy [amu*nm²/fs²] to Reporting Unit [kJ/mol]
INTERNAL_ENERGY_TO_KJ_PER_MOL = 1.0e-6
# Factor to convert Force [kJ/mol/nm] / Mass [amu] -> Acceleration [nm/fs²]
FORCE_MASS_TO_ACCEL = 1.0e-6

class Simulation:
    """
    Manages atomistic simulation using Velocity Verlet.

    Calculates forces in kJ/mol/nm and applies conversion factor only during
    acceleration calculation in the integrator step (F/m * FACTOR).
    Uses Maxwell-Boltzmann initialization for realistic starting velocities.
    Relies on physical potentials and appropriate timestep (dt_fs) for stability.

    Internal Units: Time(fs), Length(nm), Mass(amu), Velocity(nm/fs).
    Force Calculation Unit: kJ/mol/nm.
    Acceleration Unit: nm/fs².
    Energy Reporting: kJ/mol. Temperature Reporting: K.
    """
    def __init__(
        self,
        dt_fs, total_time_fs, interaction_params, target_temperature,
        rescale_interval, well_instance, r0_OH, theta0_HOH,
        k_stretch_kj_nm2, k_bend_kj_rad2, charge_O, charge_H,
        k_B_kj, k_e_kj
        # Note: initial_velocity_magnitude is intentionally removed
    ):
        # --- Essential Simulation Parameters ---
        self.dt_fs = dt_fs
        self.total_time_fs = total_time_fs
        self.target_temperature = target_temperature # K
        self.rescale_interval = rescale_interval # steps
        self.well = well_instance # nm

        # --- Force Field Parameters (kJ/mol based units) ---
        self.interaction_params = interaction_params # {eps[kJ/mol], sigma[nm], cutoff[nm]}
        self.r0_OH = r0_OH                # nm
        self.theta0_HOH = theta0_HOH      # radians
        self.k_stretch_kj_nm2 = k_stretch_kj_nm2 # kJ/(mol*nm²)
        self.k_bend_kj_rad2 = k_bend_kj_rad2   # kJ/(mol*rad²)
        self.charge_O = charge_O          # e
        self.charge_H = charge_H          # e
        self.k_B_kj = k_B_kj              # kJ/(mol*K)
        self.k_e_kj = k_e_kj              # kJ*nm/(mol*e²)

        # --- State Variables ---
        self.current_step = 0
        self.balls = [] # List of Ball objects
        self.molecules = {} # Dict mapping molecule_id to atom indices {'mol_id': {'O':idx, 'H1':idx, 'H2':idx}}
        self.temperature_history = [] # List of temperatures (K)
        self.potential_energy_history = [] # List of potential energies (kJ/mol)
        self.acceptance_rate_mc = [] # For optional MC moves

        # --- Validation ---
        if self.dt_fs <= 0: raise ValueError("dt_fs must be positive.")
        if self.k_stretch_kj_nm2 < 0 or self.k_bend_kj_rad2 < 0: raise ValueError("Force constants non-negative.")
        if self.dt_fs > 2.0: warnings.warn(f"dt ({self.dt_fs} fs) may be large. Consider <= 1.0 fs.", UserWarning)


    @classmethod
    def from_config(cls, config):
        """ Factory method to create Simulation instance from config dictionary. """
        # --- Unit Conversions ---
        angstrom_to_nm = 0.1
        k_bond_area_conversion = 100.0 # (kJ/mol/Å²) -> (kJ/mol/nm²)

        # --- Extract Config Sections ---
        sim_cfg = config["simulation"]; well_cfg = config["well"]; ball_cfg = config["ball"]
        interaction_cfg = config["interaction_params"]; particles_cfg = config["particles"]

        # --- Build Well ---
        well = Well( radius=well_cfg["radius"] * angstrom_to_nm, height=well_cfg["height"] * angstrom_to_nm,
                     wall_decay_length=well_cfg.get("wall_decay_length", 0.1) * angstrom_to_nm,
                     repulsion_constant=well_cfg.get("repulsion_constant", 2000.0), # Assumed scales force kJ/mol/nm
                     atom_radius=well_cfg.get("atom_radius", 0.1) * angstrom_to_nm )

        # --- Process Config Parameters for __init__ ---
        r0_OH_nm = ball_cfg['bond_length'] * angstrom_to_nm
        theta0_HOH_rad = np.radians(ball_cfg['angle_degrees'])
        k_stretch_kj_nm2 = ball_cfg['k_bond'] * k_bond_area_conversion # kJ/(mol*nm²)
        k_bend_kj_rad2 = ball_cfg['k_angle'] # kJ/(mol*rad²)
        charge_O = ball_cfg['oxygen_charge']; charge_H = ball_cfg['hydrogen_charge']
        interaction_params = {}
        for key, params in interaction_cfg.items():
            interaction_params[key] = params.copy()
            if 'sigma' in params: interaction_params[key]['sigma'] *= angstrom_to_nm
            if 'cutoff' in params: interaction_params[key]['cutoff'] *= angstrom_to_nm
        k_B_kj = sim_cfg['boltzmann_constant']; k_e_kj = sim_cfg['coulomb_constant']
        dt_fs = sim_cfg["dt"]; total_time_fs = sim_cfg["total_time"]
        target_temperature = sim_cfg["target_temperature"]
        rescale_interval = sim_cfg["rescale_interval"]

        # --- Create Simulation Instance (No velocities assigned yet) ---
        sim = cls( dt_fs=dt_fs, total_time_fs=total_time_fs, interaction_params=interaction_params,
                   target_temperature=target_temperature, rescale_interval=rescale_interval, well_instance=well,
                   r0_OH=r0_OH_nm, theta0_HOH=theta0_HOH_rad, k_stretch_kj_nm2=k_stretch_kj_nm2,
                   k_bend_kj_rad2=k_bend_kj_rad2, charge_O=charge_O, charge_H=charge_H, k_B_kj=k_B_kj,
                   k_e_kj=k_e_kj ) # NO initial_velocity_magnitude

        # --- Create and Add Balls Temporarily (with zero initial velocity) ---
        temp_balls_data = [] # Store tuples: (ball_instance, molecule_id, species)
        mass_O = ball_cfg['oxygen_mass']; mass_H = ball_cfg['hydrogen_mass']
        oxygen_args = {'mass': mass_O, 'charge': charge_O, 'color': ball_cfg['oxygen_color'], 'size': ball_cfg['oxygen_size']}
        hydrogen_args = {'mass': mass_H, 'charge': charge_H, 'color': ball_cfg['hydrogen_color'], 'size': ball_cfg['hydrogen_size']}
        common_defaults = {'bond_length': sim.r0_OH, 'angle_degrees': np.degrees(sim.theta0_HOH)}
        oxygen_defaults = {**oxygen_args, **common_defaults}; hydrogen_defaults = {**hydrogen_args, **common_defaults}

        for molecule_cfg in particles_cfg.get("oxygen_molecules", []):
            center_pos_nm = np.array(molecule_cfg["center_position"]) * angstrom_to_nm
            h1_z_abs_nm = molecule_cfg["H1_z"] * angstrom_to_nm; h2_z_abs_nm = molecule_cfg["H2_z"] * angstrom_to_nm
            molecule_id = molecule_cfg["molecule_id"]; zero_vel = np.zeros(3)
            oxygen = Ball.create_oxygen(center_pos_nm, zero_vel, molecule_id, defaults=oxygen_defaults)
            h1 = Ball.create_hydrogen(center_pos_nm, h1_z_abs_nm, zero_vel, molecule_id, sign=+1, defaults=hydrogen_defaults)
            h2 = Ball.create_hydrogen(center_pos_nm, h2_z_abs_nm, zero_vel, molecule_id, sign=-1, defaults=hydrogen_defaults)
            temp_balls_data.append((oxygen, molecule_id, "O")); temp_balls_data.append((h1, molecule_id, "H1")); temp_balls_data.append((h2, molecule_id, "H2"))

        # --- Assign Initial Velocities from Maxwell-Boltzmann ---
        if target_temperature > 0 and k_B_kj > 0 and temp_balls_data:
            print(f"Assigning initial velocities for T={target_temperature}K")
            # v_sigma = sqrt(k_B[kJ/mol/K] * T[K] / mass[amu]) * 1e-3 -> nm/fs
            conv_factor_v = 1.0e-3
            for ball_data in temp_balls_data:
                ball, _, _ = ball_data
                if ball.mass > 0:
                    v_sigma = np.sqrt(k_B_kj * target_temperature / ball.mass) * conv_factor_v # nm/fs
                    ball.velocity = np.random.normal(loc=0.0, scale=v_sigma, size=3) # nm/fs
        else:
             print("Warning: Initial velocities set to zero (T=0 or no particles).")

        # --- Finalize Ball List and Molecule Mapping ---
        mol_map = {}
        for ball_data in temp_balls_data:
            ball, mol_id, species_label = ball_data
            idx = sim.add_ball(ball)
            if mol_id:
                if mol_id not in mol_map: mol_map[mol_id] = {}
                mol_map[mol_id][species_label] = idx # Map O, H1, H2 correctly
        sim.molecules = mol_map

        sim.remove_com_velocity() # Remove drift after random velocities assigned
        # Optional: Apply thermostat once immediately to precisely match target T initially
        # sim.apply_velocity_rescaling()
        # print(f"Initial Temperature after velocity assignment: {sim.compute_system_temperature():.2f} K") # Debug print

        return sim


    def add_ball(self, ball):
        """ Adds a ball to the simulation list and returns its index. """
        idx = len(self.balls); self.balls.append(ball); return idx

    # create_water_molecule is primarily conceptual now, used by from_config


    def compute_forces(self):
        """ Calculates forces on all balls in kJ/mol/nm. """
        num_balls = len(self.balls); [ball.force.fill(0.0) for ball in self.balls]

        # 1. Intramolecular forces (kJ/mol/nm)
        if self.k_stretch_kj_nm2 > 0 or self.k_bend_kj_rad2 > 0:
            for indices in self.molecules.values(): # Assumes indices are O, H1, H2
                # Error check: Ensure molecule definition is complete before accessing
                if "O" not in indices or "H1" not in indices or "H2" not in indices: continue
                O, H1, H2 = self.balls[indices["O"]], self.balls[indices["H1"]], self.balls[indices["H2"]]
                # Bond Stretching
                if self.k_stretch_kj_nm2 > 0:
                    dOH1=H1.position-O.position; rOH1=np.linalg.norm(dOH1)
                    if rOH1!=0: fOH1=(-self.k_stretch_kj_nm2*(rOH1-self.r0_OH)/rOH1)*dOH1; H1.force+=fOH1; O.force-=fOH1
                    dOH2=H2.position-O.position; rOH2=np.linalg.norm(dOH2)
                    if rOH2!=0: fOH2=(-self.k_stretch_kj_nm2*(rOH2-self.r0_OH)/rOH2)*dOH2; H2.force+=fOH2; O.force-=fOH2
                # Angle Bending
                if self.k_bend_kj_rad2 > 0:
                    vOH1=H1.position-O.position; vOH2=H2.position-O.position; nOH1=np.linalg.norm(vOH1); nOH2=np.linalg.norm(vOH2)
                    if nOH1!=0 and nOH2!=0: # Prevent div by zero
                        ct=np.clip(np.dot(vOH1,vOH2)/(nOH1*nOH2),-1,1); th=np.arccos(ct); st=np.sqrt(max(0.0, 1.0-ct**2)) # Use max for stability
                        if st!=0: # Prevent div by zero for collinear
                            tq=-self.k_bend_kj_rad2*(th-self.theta0_HOH)/st; tH1=tq/nOH1; tH2=tq/nOH2
                            uOH1=vOH1/nOH1; uOH2=vOH2/nOH2; FH1=tH1*(ct*uOH1-uOH2); FH2=tH2*(ct*uOH2-uOH1); FO=-(FH1+FH2)
                            H1.force+=FH1; H2.force+=FH2; O.force+=FO

        # 2. Intermolecular forces (kJ/mol/nm)
        cut_sq_def=self.interaction_params.get("default",{}).get("cutoff",np.inf)**2; cut_sq={k:p.get("cutoff",np.inf)**2 for k,p in self.interaction_params.items()}
        for i in range(num_balls):
            bi=self.balls[i]; qi=bi.charge
            for j in range(i + 1, num_balls): # Pair loop
                bj=self.balls[j]
                # Skip intramolecular non-bonded
                if bi.molecule_id is not None and bi.molecule_id == bj.molecule_id: continue
                # Distance vector + PBC
                delta=bi.position-bj.position; delta[2]-=self.well.height*np.round(delta[2]/self.well.height)
                r_sq=np.dot(delta,delta)
                # Check cutoff & non-overlap
                spk="-".join(sorted([bi.species,bj.species])); cutoff_sq=cut_sq.get(spk, cut_sq_def)
                if r_sq < cutoff_sq and r_sq != 0:
                    r=np.sqrt(r_sq); du=delta/r; f_nb=np.zeros(3)
                    # LJ Force
                    ljp=self.interaction_params.get(spk, self.interaction_params.get("default"))
                    if ljp: eps=ljp.get("epsilon",0); sig=ljp.get("sigma",0)
                    if eps>0 and sig>0: sr=sig/r; sr6=sr**6; ljm=(24*eps/r)*(2*sr6**2-sr6); f_nb+=ljm*du
                    # Coulomb Force
                    qj=bj.charge
                    if qi!=0 and qj!=0: clm=(self.k_e_kj*qi*qj/r_sq); f_nb+=clm*du
                    # Apply force
                    bi.force+=f_nb; bj.force-=f_nb

        # 3. Wall Repulsion forces (kJ/mol/nm)
        for ball in self.balls:
            ball.force += self.well.compute_wall_repulsion_force(ball) # Expects kJ/mol/nm


    def update(self):
        """ Advances simulation by dt_fs using Velocity Verlet. """
        # --- VV Step 1 & 2: Update v(t+dt/2), x(t+dt) ---
        for ball in self.balls:
            # Check mass to prevent division by zero during acceleration calculation
            if ball.mass == 0: continue # Skip particles with zero mass
            acc = (ball.force / ball.mass) * FORCE_MASS_TO_ACCEL # a = F/m * factor -> nm/fs²
            ball.velocity += 0.5 * acc * self.dt_fs # v(t+dt/2)
            ball.skip_path_update = False
            ball.position += ball.velocity * self.dt_fs # x(t+dt)
            # --- Boundaries ---
            if ball.position[2]<0 or ball.position[2]>=self.well.height: ball.position[2]%=self.well.height; ball.skip_path_update=True # Z PBC
            r_sq=ball.position[0]**2+ball.position[1]**2 # XY Reflect
            if r_sq > self.well.radius**2:
                r=np.sqrt(r_sq); f=self.well.radius/r; ball.position[0]*=f; ball.position[1]*=f
                if r!=0: # Reflect velocity if not on axis
                    nx=ball.position[0]/self.well.radius; ny=ball.position[1]/self.well.radius
                    vd=ball.velocity[0]*nx + ball.velocity[1]*ny
                    if vd>0: ball.velocity[0]-=2*vd*nx; ball.velocity[1]-=2*vd*ny
                ball.skip_path_update=True

        # --- VV Step 3: Recalculate forces F(t+dt) ---
        self.compute_forces() # Calculate F(t+dt) in kJ/mol/nm

        # --- VV Step 4: Update velocities v(t+dt) ---
        for ball in self.balls:
            if ball.mass == 0: continue # Skip particles with zero mass
            acc = (ball.force / ball.mass) * FORCE_MASS_TO_ACCEL # a(t+dt) -> nm/fs²
            ball.velocity += 0.5 * acc * self.dt_fs # v(t+dt)

        # --- Thermostat (Less Frequent) ---
        if self.rescale_interval > 0 and self.current_step > 0 and (self.current_step % self.rescale_interval == 0):
            self.apply_velocity_rescaling()

        # --- Recording & Step Increment ---
        self.temperature_history.append(self.compute_system_temperature())
        # Optionally calculate PE here
        # self.potential_energy_history.append(self.compute_total_potential_energy())
        for ball in self.balls: ball.update_path()
        self.current_step += 1


    def compute_system_temperature(self):
        """ Calculates instantaneous kinetic temperature in K. """
        if not self.balls: return 0.0;
        # KE[amu*nm²/fs²] -> KE[kJ/mol]
        tke_kj = sum(0.5 * b.mass * np.dot(b.velocity, b.velocity) for b in self.balls) * INTERNAL_ENERGY_TO_KJ_PER_MOL
        n_atoms_with_mass = sum(1 for b in self.balls if b.mass > 0) # Count only atoms with mass
        if n_atoms_with_mass == 0: return 0.0
        degrees_of_freedom = 3 * n_atoms_with_mass # Use DOF based on atoms that can move
        if self.k_B_kj == 0: return 0.0
        temperature = (2.0 * tke_kj) / (degrees_of_freedom * self.k_B_kj) # T = 2*KE / (Ndf*kB)
        # Handle potential negative temperatures from numerical noise at low T
        return max(0.0, temperature) # Ensure temperature is non-negative

    def apply_velocity_rescaling(self):
        """ Applies Berendsen thermostat velocity scaling. """
        current_temp = self.compute_system_temperature()
        # Avoid division by zero or sqrt of negative if T is zero or target T is negative
        if current_temp <= 1e-9 or self.target_temperature <= 0: return # Add back a small check for current_temp
        scale_factor = np.sqrt(self.target_temperature / current_temp)
        for ball in self.balls: ball.velocity *= scale_factor

    def remove_com_velocity(self):
        """ Subtracts center-of-mass velocity. """
        movable_balls = [b for b in self.balls if b.mass > 0]
        if not movable_balls: return
        total_mass = sum(b.mass for b in movable_balls)
        if total_mass == 0: return # Should not happen if movable_balls is not empty
        total_momentum = np.sum([b.mass * b.velocity for b in movable_balls], axis=0) # amu*nm/fs
        com_velocity = total_momentum / total_mass # nm/fs
        # Apply only to balls that can move
        for ball in movable_balls: ball.velocity -= com_velocity


    # --- compute_total_potential_energy (Corrected Indentation) ---
    # In simulation.py

    def compute_total_potential_energy(self):
        """ Computes total potential energy in kJ/mol. """
        pe_bond_angle_kj = 0.0  # Accumulate kJ/mol for intramolecular
        pe_nonbond_kj = 0.0  # Accumulate kJ/mol for intermolecular

        # 1. Intramolecular PE (Units: kJ/mol)
        # Check if intramolecular forces are active
        if self.k_stretch_kj_nm2 > 0 or self.k_bend_kj_rad2 > 0:
            # Loop through EACH molecule
            for molecule_id, indices in self.molecules.items():
                # Get the Ball objects for this molecule
                O = self.balls[indices["O"]]
                H1 = self.balls[indices["H1"]]
                H2 = self.balls[indices["H2"]]

                # Bond PE: U = 0.5 * k_kj_nm2 * (r - r0)² -> kJ/mol
                if self.k_stretch_kj_nm2 > 0 and self.r0_OH > 0:
                    delta_OH1 = H1.position - O.position
                    r_OH1 = np.linalg.norm(delta_OH1)
                    pe_bond_angle_kj += 0.5 * self.k_stretch_kj_nm2 * (r_OH1 - self.r0_OH) ** 2

                    delta_OH2 = H2.position - O.position
                    r_OH2 = np.linalg.norm(delta_OH2)
                    pe_bond_angle_kj += 0.5 * self.k_stretch_kj_nm2 * (r_OH2 - self.r0_OH) ** 2

                # Angle PE: U = 0.5 * k_kj_rad2 * (theta - theta0)² -> kJ/mol
                if self.k_bend_kj_rad2 > 0:
                    vec_OH1 = H1.position - O.position
                    vec_OH2 = H2.position - O.position
                    norm_OH1 = np.linalg.norm(vec_OH1)
                    norm_OH2 = np.linalg.norm(vec_OH2)
                    # Ensure norms are non-zero before calculating angle
                    if norm_OH1 != 0 and norm_OH2 != 0:
                        dot_p = np.dot(vec_OH1, vec_OH2)
                        cos_t = np.clip(dot_p / (norm_OH1 * norm_OH2), -1.0, 1.0)
                        theta = np.arccos(cos_t)  # radians
                        pe_bond_angle_kj += 0.5 * self.k_bend_kj_rad2 * (theta - self.theta0_HOH) ** 2
        # --- End of loop through molecules ---

        # 2. Intermolecular PE (Already kJ/mol)
        num_balls = len(self.balls)
        cutoff_sq_default = self.interaction_params.get("default", {}).get("cutoff", np.inf) ** 2
        interaction_cutoff_sq = {k: p.get("cutoff", np.inf) ** 2 for k, p in self.interaction_params.items()}

        # Loop through unique pairs of atoms
        for i in range(num_balls):
            ball_i = self.balls[i]
            q_i = ball_i.charge
            for j in range(i + 1, num_balls):
                ball_j = self.balls[j]

                # Skip pairs within the same molecule
                if ball_i.molecule_id is not None and ball_i.molecule_id == ball_j.molecule_id:
                    continue

                # Calculate distance with PBC
                delta = ball_i.position - ball_j.position
                delta[2] -= self.well.height * np.round(delta[2] / self.well.height)
                r_sq = np.dot(delta, delta)

                # Determine interaction type and cutoff
                species_key = "-".join(sorted([ball_i.species, ball_j.species]))
                cutoff_sq = interaction_cutoff_sq.get(species_key, cutoff_sq_default)

                # Calculate potential if within cutoff and not exactly overlapping
                if r_sq < cutoff_sq and r_sq != 0:
                    r = np.sqrt(r_sq)

                    # LJ PE (kJ/mol)
                    ljp = self.interaction_params.get(species_key, self.interaction_params.get("default"))
                    if ljp:
                        eps = ljp.get("epsilon", 0)
                        sig = ljp.get("sigma", 0)
                        if eps > 0 and sig > 0:
                            sr = sig / r
                            sr6 = sr ** 6
                            pe_nonbond_kj += 4.0 * eps * (sr6 ** 2 - sr6)  # U = 4*eps*((sigma/r)^12 - (sigma/r)^6)

                    # Coulomb PE (kJ/mol)
                    q_j = ball_j.charge
                    if q_i != 0 and q_j != 0:
                        pe_nonbond_kj += self.k_e_kj * q_i * q_j / r  # U = k_e * q_i * q_j / r

        # 3. Wall PE (Optional - Requires Well method returning kJ/mol)
        # pe_wall_kj = sum(self.well.compute_wall_potential_kj_mol(ball) for ball in self.balls if hasattr(self.well, 'compute_wall_potential_kj_mol'))
        # pe_nonbond_kj += pe_wall_kj

        # Return sum of intramolecular and intermolecular potential energies
        return pe_bond_angle_kj + pe_nonbond_kj


    # --- perform_monte_carlo_move (Corrected Assignment on Reject) ---
    def perform_monte_carlo_move(self,max_disp_nm=0.01,temperature_mc=None):
        if not self.balls: return; temp=temperature_mc if temperature_mc is not None else self.target_temperature
        old_pos=[b.position.copy() for b in self.balls]; old_E=self.compute_total_potential_energy()
        for b in self.balls: b.position+=np.random.uniform(-max_disp_nm,max_disp_nm,3); b.position[2]%=self.well.height; rsq=b.position[0]**2+b.position[1]**2
        if rsq>self.well.radius**2: f=self.well.radius/np.sqrt(rsq); b.position[0]*=f; b.position[1]*=f
        new_E=self.compute_total_potential_energy(); dE=new_E-old_E; accept=(dE<=0)
        if not accept and temp>0 and self.k_B_kj>0: bf=np.exp(-dE/(self.k_B_kj*temp)); accept=(random.random()<bf)
        if accept: self.potential_energy_history.append(new_E); [b.update_path() for b in self.balls]; self.acceptance_rate_mc.append(1)
        else:
            # Correct way to restore positions
            for i, ball in enumerate(self.balls): ball.position = old_pos[i]
            self.potential_energy_history.append(old_E); self.acceptance_rate_mc.append(0)
        self.temperature_history.append(self.compute_system_temperature()); self.current_step+=1