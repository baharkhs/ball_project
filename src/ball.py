# ball.py
import numpy as np

INTERNAL_ENERGY_TO_KJ_PER_MOL = 1.0e-6

class Ball:
    """ Represents a single particle (atom) in the simulation. """
    def __init__(self, species, position, velocity, molecule_id=None, mass=1.0, charge=0.0, color="gray", size=8):
        self.species = species
        self.position = np.array(position, dtype=float) # nm
        self.velocity = np.array(velocity, dtype=float) # nm/fs
        self.molecule_id = molecule_id
        if mass <= 0: raise ValueError(f"Mass must be positive for species {species}.")
        self.mass = mass             # amu
        self.charge = charge         # e
        self.color = color
        self.size = size
        # Force vector, calculated in kJ/mol/nm by Simulation.compute_forces
        self.force = np.zeros(3)
        # Path tracking
        self.path_segments = []
        self.current_path_segment = {"x": [], "y": [], "z": []}
        self.skip_path_update = False

    @classmethod
    def create_oxygen(cls, position, velocity, molecule_id, defaults):
        mass = defaults.get("oxygen_mass", 15.999); charge = defaults.get("oxygen_charge", -0.8476)
        color = defaults.get("oxygen_color", "blue"); size = defaults.get("oxygen_size", 10)
        if mass <= 0: raise ValueError("Oxygen mass must be positive.")
        return cls("O", position, velocity, molecule_id, mass=mass, charge=charge, color=color, size=size)

    @classmethod
    def create_hydrogen(cls, o_position_nm, h_z_abs_nm, velocity, molecule_id, sign, defaults):
        """ Creates Hydrogen, robustly handling geometry hints. """
        bond_length_nm = defaults.get("bond_length", 0.1) # nm
        angle_degrees = defaults.get("angle_degrees", 109.47) # degrees
        if bond_length_nm <= 0: raise ValueError("Bond length must be positive.")
        half_angle_rad = np.radians(angle_degrees) / 2.0 # radians

        dz = h_z_abs_nm - o_position_nm[2] # Relative Z hint

        # --- Robust Geometry Calculation ---
        # Ensure |dz| <= bond_length to avoid sqrt(negative)
        if abs(dz) > bond_length_nm:
             # If hint is impossible, place H in XY plane relative to O for simplicity
             # This is arbitrary but prevents NaN. Simulation forces will correct it.
             # print(f"Warning: Z-hint {h_z_abs_nm} for {molecule_id} impossible with bond length {bond_length_nm}. Placing in XY plane.")
             dz = 0.0 # Override dz

        proj_len_sq = bond_length_nm**2 - dz**2
        # Should not be negative now, but handle floating point cases just in case
        proj_len = np.sqrt(max(0.0, proj_len_sq)) # Length projection in XY plane

        # Determine XY components
        # Place symmetrically around the local Y-axis (arbitrary but consistent)
        dx = sign * proj_len * np.sin(half_angle_rad)
        dy = proj_len * np.cos(half_angle_rad)
        # --- End Robust Geometry ---

        disp_vec = np.array([dx, dy, dz])
        h_position_nm = o_position_nm + disp_vec # Initial H position

        # Retrieve other properties
        mass = defaults.get("hydrogen_mass", 1.008); charge = defaults.get("hydrogen_charge", 0.4238)
        color = defaults.get("hydrogen_color", "red"); size = defaults.get("hydrogen_size", 6) # Reverted size for clarity
        if mass <= 0: raise ValueError("Hydrogen mass must be positive.")
        return cls("H", h_position_nm, velocity, molecule_id, mass=mass, charge=charge, color=color, size=size)

    def calculate_kinetic_energy(self):
        """ Calculates kinetic energy in kJ/mol. """
        ke_internal = 0.5 * self.mass * np.dot(self.velocity, self.velocity) # amu*nm²/fs²
        ke_kj_mol = ke_internal * INTERNAL_ENERGY_TO_KJ_PER_MOL # Convert to kJ/mol
        return ke_kj_mol

    # --- Path Tracking Methods (Unchanged) ---
    def update_path(self):
        if self.skip_path_update:
            if self.current_path_segment["x"]: self.path_segments.append(self.current_path_segment)
            self.current_path_segment = { "x": [self.position[0]], "y": [self.position[1]], "z": [self.position[2]] }
            self.skip_path_update = False
        else:
            self.current_path_segment["x"].append(self.position[0])
            self.current_path_segment["y"].append(self.position[1])
            self.current_path_segment["z"].append(self.position[2])

    def finalize_path(self):
        if self.current_path_segment["x"]:
            self.path_segments.append(self.current_path_segment)
            self.current_path_segment = {"x": [], "y": [], "z": []}

    def get_path_segments(self):
        all_segments = self.path_segments
        if self.current_path_segment["x"]:
             all_segments = self.path_segments + [self.current_path_segment.copy()]
        return all_segments