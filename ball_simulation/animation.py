import numpy as np

class Well:
    def __init__(self, radius=5.0, height=10.0, cutoff_radius=None):
        """
        Initializes the Well object representing a cylindrical space for particle movement.

        Args:
            radius (float): Radius of the cylindrical well (in angstroms).
            height (float): Height of the well (in angstroms).
            cutoff_radius (float): Optional cutoff radius for smoothing forces (if not provided, defaults to radius).

        The 'radius' and 'height' define the geometry of the confined area:
        - The well extends from z=0 to z=height.
        - The radial boundary in the x-y plane is given by radius.

        If cutoff_radius is provided, it can be used to smoothly reduce forces near the boundary.
        Otherwise, cutoff_radius defaults to the well radius.
        """
        self.radius = radius
        self.height = height
        # If no cutoff is given, use the well radius. This will be used for smoothing force application near walls.
        self.cutoff_radius = cutoff_radius if cutoff_radius else radius

    def compute_wall_repulsion_force(self, ball, repulsion_constant=1.0):
        """
        Compute repulsion from the well boundaries.

        Args:
            ball (Ball): The ball object, which should have position and radius attributes.
            repulsion_constant (float): Strength of the repulsion.

        Returns:
            np.array: A 3D force vector acting on the ball from the well boundaries.

        Explanation:
        - We consider repulsive forces from:
          1. The cylindrical x–y boundary (radial direction).
          2. The top and bottom walls (if the ball tries to go below z=0 or above z=height).
        """
        force = np.zeros(3)
        distance_from_center = np.linalg.norm(ball.position[:2])

        # Radial boundary
        if distance_from_center + ball.radius > self.radius:
            overlap = distance_from_center + ball.radius - self.radius
            force[:2] = -overlap * (ball.position[:2] / distance_from_center)

        # Bottom boundary
        if ball.position[2] - ball.radius < 0:
            overlap = ball.radius - ball.position[2]
            force[2] += overlap

        # Top boundary
        elif ball.position[2] + ball.radius > self.height:
            overlap = ball.position[2] + ball.radius - self.height
            force[2] -= overlap

        return force

    def apply_pbc(self, position):
        """
        Applies periodic boundary conditions (PBC) along the z-axis only.

        Args:
            position (np.array): The current position [x, y, z] of the particle.

        Returns:
            tuple: (modified_position, wrapped)
                modified_position: Position after applying PBC if needed.
                wrapped (bool): True if a wrap occurred, False otherwise.

        Explanation:
        - If the particle goes above z=height, it reappears at z -= height.
        - If it goes below z=0, it reappears at z += height.

        Physically, this simulates a system that is "infinite" along z by wrapping around,
        similar to imagining a particle leaving the top and entering from the bottom.
        """
        wrapped = False
        if position[2] > self.height:
            position[2] -= self.height
            wrapped = True
        elif position[2] < 0:
            position[2] += self.height
            wrapped = True
        return position, wrapped

    def apply_bounce(self, ball):
        """
        Applies a rigid bounce if the particle reaches the x–y boundary.

        Args:
            ball (Ball): The Ball object with attributes like position and velocity.

        Returns:
            np.array: The updated velocity after a bounce if applied.

        Explanation:
        - This method checks if the ball is outside or exactly on the boundary in x–y plane.
        - If it is, the velocity is reflected about the normal vector to the wall.
        - This is a hard, instantaneous "bounce", which is non-physical because real walls would
          apply a smoothly increasing repulsive force before the particle reaches the boundary.

        While simple, this can cause numerical issues and unphysical behavior. It's here as a
        fallback or a simple boundary condition. For more realism, rely on smooth repulsive forces.
        """
        distance_from_center = np.linalg.norm(ball.position[:2])
        # Check if the particle is at or beyond the radial boundary.
        if distance_from_center >= self.radius:
            normal_direction = ball.position[:2] / distance_from_center
            # Reflect the velocity component normal to the wall:
            # v_new = v_old - 2*(v_old·n)*n
            ball.velocity[:2] -= 2 * np.dot(ball.velocity[:2], normal_direction) * normal_direction
            # Push the particle slightly inside to avoid it getting stuck exactly at boundary.
            ball.position[:2] = normal_direction * (self.radius - 1e-10)
        return ball.velocity

    def smooth_cutoff_function(self, r, cutoff):
        """
        Computes a smooth scaling factor that tapers off as distance approaches the cutoff.

        Args:
            r (float): The current distance measure (e.g., distance from boundary or zero point).
            cutoff (float): The cutoff distance at which the force goes to zero.

        Returns:
            float: A scaling factor in [0,1] that reduces the force to zero at r=cutoff.

        Explanation:
        - If r > cutoff, return 0: no force beyond that range.
        - If r <= cutoff, use a polynomial decay: (1 - (r/cutoff)^2)^2.

        Currently, this reduces force to zero at the cutoff. In some physical scenarios, you might
        want the force to increase as you approach the wall. If so, you could invert the logic.
        For now, we keep the code as is to maintain original features.
        """
        if r > cutoff:
            return 0.0
        # Smooth polynomial decay factor
        return (1 - (r / cutoff) ** 2) ** 2

    def compute_wall_repulsion_force(self, ball, repulsion_constant=1.0):
        """
        Computes a repulsive force from the well walls.

        Args:
            ball (Ball): The ball object, which should have position and radius attributes.
            repulsion_constant (float): Strength of the repulsion.

        Returns:
            np.array: A 3D force vector acting on the ball from the well boundaries.

        Explanation:
        - We consider repulsive forces from:
          1. The cylindrical x–y boundary (radial direction).
          2. The top and bottom walls (if the ball tries to go below z=0 or above z=height).

        - For the radial boundary:
            We check how close the particle is to the boundary (distance_from_center to radius).
            If the particle "overlaps" with the boundary (i.e., ball radius extends beyond the well radius),
            we compute a repulsive force pushing it back inside.

          overlap = ball.radius - (self.radius - distance_from_center)
          If overlap > 0, the particle is beyond the allowed boundary. We push it back inward.
          To keep original structure, we use the smooth_cutoff_function, but note that right now it
          reduces force near cutoff. For a well, you might want it to be very large near the boundary.

        - For the z-boundaries:
            If ball.position[2] < ball.radius, it means the ball extends below the bottom of the well.
            Similarly, if ball.position[2] > self.height - ball.radius, it extends above the top.
            In both cases, we compute an overlap and push the ball back in.

        - The repulsion_constant sets how strong the repulsion is.
        - The smooth_factor scales the force down as we approach the cutoff radius.

        NOTE:
        If we are truly using PBC along z, having a top and bottom boundary is physically contradictory.
        Either remove PBC and treat top/bottom similarly to the sides, or remove top/bottom forces if
        you want an infinite cylinder. For now, we keep it to maintain original structure.
        """
        force = np.zeros(3)

        # Compute radial distance in x-y plane
        distance_from_center = np.linalg.norm(ball.position[:2])

        # A small numerical safeguard in case distance_from_center is extremely close to zero:
        if distance_from_center < 1e-14:
            distance_from_center = 0.0

        # Compute overlap for radial boundary
        if distance_from_center > 0:
            overlap = ball.radius - (self.radius - distance_from_center)
        else:
            # If the ball is exactly at center, consider no overlap unless ball.radius > self.radius
            overlap = ball.radius - self.radius if ball.radius > self.radius else - (self.radius - ball.radius)

        if overlap > 0:
            normal_direction = ball.position[:2] / distance_from_center if distance_from_center > 0 else np.array([0.0, 0.0])
            repulsion_magnitude = repulsion_constant * overlap
            smooth_factor = self.smooth_cutoff_function(distance_from_center, self.cutoff_radius)
            force[:2] = -repulsion_magnitude * smooth_factor * normal_direction

        # Bottom wall
        if ball.position[2] < ball.radius:
            z_overlap = ball.radius - ball.position[2]
            if z_overlap > 0:
                smooth_factor_z = self.smooth_cutoff_function(ball.position[2], ball.radius)
                force[2] += repulsion_constant * z_overlap * smooth_factor_z

        # Top wall
        elif ball.position[2] > self.height - ball.radius:
            z_overlap = ball.radius - (self.height - ball.position[2])
            if z_overlap > 0:
                dist_from_top = self.height - ball.position[2]
                smooth_factor_z = self.smooth_cutoff_function(dist_from_top, ball.radius)
                force[2] -= repulsion_constant * z_overlap * smooth_factor_z

        return force

    def compute_total_wall_force(self, ball, repulsion_constant=1.0):
        """
        Computes the total force from the well boundaries on the ball.

        Args:
            ball (Ball): The ball object.
            repulsion_constant (float): Strength of the repulsion force.

        Returns:
            np.array: Total wall force acting on the ball.

        Explanation:
        - Currently, this just calls compute_wall_repulsion_force.
        - This method can be extended if you want to add additional effects, such as friction or
          other boundary interactions.
        """
        repulsion_force = self.compute_wall_repulsion_force(ball, repulsion_constant)
        return repulsion_force


class Ball:
    def __init__(self, position, velocity, mass, radius, species="unknown", attraction_coeff=0.0, repulsion_coeff=0.0):
        """
        Initialize a Ball object.

        Args:
            position (array-like): Initial position of the ball [x, y, z].
            velocity (array-like): Initial velocity of the ball [vx, vy, vz].
            mass (float): Mass of the ball.
            radius (float): Radius of the ball.
            species (str): Species of the ball (e.g., "H", "O").
            attraction_coeff (float): Coefficient for attraction forces.
            repulsion_coeff (float): Coefficient for repulsion forces.
        """
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass
        self.radius = radius
        self.species = species
        self.attraction_coeff = attraction_coeff
        self.repulsion_coeff = repulsion_coeff

        self.force = np.zeros(3)
        self.path_segments = []
        self.current_path_segment = {"x": [], "y": [], "z": []}

    def compute_force(self, other_balls):
        """
        Compute the net force on this ball due to attraction and repulsion from other balls.

        Args:
            other_balls (list): List of other Ball objects.
        """
        self.force[:] = 0.0  # Reset force

        for other in other_balls:
            if other is self:
                continue

            r_vec = other.position - self.position
            r_mag = np.linalg.norm(r_vec)

            if r_mag == 0:
                continue  # Avoid singularity

            r_hat = r_vec / r_mag

            # Attraction force: F_attr = -k_a * r_hat
            F_attr = -self.attraction_coeff * r_hat

            # Repulsion force: F_rep = k_r / r_mag^2 * r_hat
            F_rep = (self.repulsion_coeff / r_mag ** 2) * r_hat

            self.force += F_attr + F_rep

    def compute_interaction_force(self, other, interaction_params):
        """
        Compute the interaction force between this ball and another ball using Lennard-Jones potential.

        Args:
            other (Ball): Another ball to calculate the force against.
            interaction_params (dict): Interaction parameters containing epsilon, sigma, and cutoff.

        Returns:
            np.array: Force vector in 3D space acting on this ball due to the other ball.
        """
        # Extract parameters for interaction
        pair_key = tuple(sorted([self.species, other.species]))
        params = interaction_params.get(pair_key, None)

        if params is None:
            raise ValueError(f"Interaction parameters for species pair {pair_key} are missing.")

        sigma = params.get("sigma", 1.0)
        epsilon = params.get("epsilon", 1.0)
        cutoff = params.get("cutoff", 5.0)

        # Compute distance vector and magnitude
        r_vec = self.position - other.position
        r_mag = np.linalg.norm(r_vec)

        if r_mag == 0 or r_mag > cutoff:
            # No force if distance is zero (avoid singularity) or beyond cutoff
            return np.zeros(3)

        # Unit vector in the direction of the force
        r_hat = r_vec / r_mag

        # Lennard-Jones force magnitude
        sr = sigma / r_mag
        sr6 = sr ** 6
        sr12 = sr6 ** 2
        force_magnitude = 48 * epsilon * ((sr12 / r_mag) - 0.5 * (sr6 / r_mag))

        # Force vector
        return force_magnitude * r_hat

    def calculate_kinetic_energy(self):
        """
        Calculate the kinetic energy of the ball.

        Formula:
        KE = 0.5 * mass * |velocity|^2

        Returns:
            float: Kinetic energy in units of amu*(Å/fs)^2.
        """
        return 0.5 * self.mass * np.dot(self.velocity, self.velocity)

    def update_position(self, dt):
        """
        Update the position and velocity of the ball using Newton's second law.

        Args:
            dt (float): Time step for the update.
        """
        acceleration = self.force / self.mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

    def update_path(self, finalize=False):
        """
        Update the particle's path by recording its current position or finalize the current segment.

        Args:
            finalize (bool): If True, finalize the current path segment and start a new one.
        """
        if finalize:
            if self.current_path_segment["x"]:
                self.path_segments.append(self.current_path_segment)
            self.current_path_segment = {"x": [], "y": [], "z": []}  # Start a new segment
        else:
            self.current_path_segment["x"].append(self.position[0])
            self.current_path_segment["y"].append(self.position[1])
            self.current_path_segment["z"].append(self.position[2])

    def get_all_paths(self):
        """
        Return all recorded path segments for visualization, including the current segment.

        Returns:
            list: A list of all path segments, finalized and ongoing.
        """
        return self.path_segments + [self.current_path_segment] if self.current_path_segment["x"] else self.path_segments


class Simulation:
    def __init__(self, well_radius=0.5, well_height=1.0, total_time=10.0, dt=0.01, movement_type="newtonian"):

        """
        Initializes the simulation environment, including the container (well), time parameters, and movement type.

        Args:
            well_radius (float): Radius of the cylindrical well in Å.
            well_height (float): Height of the well in Å.
            total_time (float): Total simulation time in fs.
            dt (float): Time step in fs.
            movement_type (str): "newtonian" (deterministic) or "monte_carlo" (stochastic) particle dynamics.

        Scientific Details:
        - Positions are in Å, velocities in Å/fs, time in fs, and mass in amu.
        - Forces are computed in amu·Å/fs², energies in amu·(Å/fs)².
        - Temperature is in Kelvin (K).

        The `movement_type` influences position updates:
        - "newtonian": Classical equations of motion (F = ma).
        - "monte_carlo": Random steps and stochastic forces.
        """
        self.well = Well(radius=well_radius, height=well_height)
        self.well_radius = well_radius
        self.well_height = well_height
        self.dt = dt
        self.total_time = total_time
        self.movement_type = movement_type
        self.balls = []  # List of Ball objects
        self.interaction_params = {
            ("O", "O"): {"epsilon": 0.5, "sigma": 0.3, "cutoff": 5.0},
            ("H", "O"): {"epsilon": 0.3, "sigma": 0.25, "cutoff": 4.0},
            ("H", "H"): {"epsilon": 0.2, "sigma": 0.2, "cutoff": 3.0},
        }
        self.thermostat_method = None

    def set_movement_type(self, movement_type="newtonian"):
        """
        Sets the movement type for the simulation.

        Args:
            movement_type (str): "newtonian" or "monte_carlo".
        """
        self.movement_type = movement_type

    def add_ball(self, ball):
        """
        Add a ball to the simulation.

        Args:
            ball (Ball): A Ball object to be added.
        """
        self.balls.append(ball)

    def apply_thermostat(self, target_temperature, method="nose_hoover", gamma=1.0):
        """
        Apply a thermostat to regulate the system's temperature.

        Args:
            target_temperature (float): Desired temperature in Kelvin.
            method (str): "nose_hoover" or "langevin".
            gamma (float): Friction coefficient for Langevin thermostat.

        Scientific Details:
        - Nose-Hoover: Enforces canonical ensemble by controlling kinetic energy fluctuations.
        - Langevin: Adds stochastic and frictional forces to mimic heat bath interactions.
        """
        current_temperature = self.calculate_average_temperature()
        scaling_factor = np.sqrt(target_temperature / current_temperature)

        if method == "nose_hoover":
            for ball in self.balls:
                ball.velocity *= scaling_factor

        elif method == "langevin":
            for ball in self.balls:
                random_force = np.random.normal(0, np.sqrt(gamma * target_temperature), size=3)
                ball.force += random_force

    def update(self):
        """
        Perform a single simulation step:
        1. Reset forces.
        2. Compute interaction forces.
        3. Apply wall forces.
        4. Update positions and velocities.
        5. Apply boundary conditions.
        6. Compute temperature and energy.

        Scientific Details:
        - Forces include Lennard-Jones interactions and wall repulsions.
        - Positions are updated using Newtonian mechanics.
        """
        self._reset_forces()
        self._compute_pairwise_forces()
        self._apply_wall_forces()
        self._update_positions_and_velocities()
        self._apply_boundary_conditions()

    def _reset_forces(self):
        """Reset forces on all balls."""
        for ball in self.balls:
            ball.force = np.zeros(3)

    def _compute_pairwise_forces(self):
        """
        Compute Lennard-Jones interaction forces between balls.

        Scientific Equation:
        F(r) = 48 * epsilon * [ (sigma^12 / r^13) - 0.5 * (sigma^6 / r^7) ] * r̂
        """
        num_balls = len(self.balls)
        for i in range(num_balls):
            for j in range(i + 1, num_balls):
                force = self.balls[i].compute_interaction_force(self.balls[j], self.interaction_params)
                self.balls[i].force += force
                self.balls[j].force -= force

    def _apply_wall_forces(self):
        """
        Compute repulsive forces from the simulation boundaries.

        Scientific Details:
        - Walls exert repulsive forces based on particle overlap.
        """
        for ball in self.balls:
            wall_force = self.well.compute_wall_repulsion_force(ball)
            ball.force += wall_force



    def _update_positions_and_velocities(self):
        """
        Update the positions and velocities of all balls.

        Newtonian Mechanics:
        x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
        v(t+dt) = v(t) + a(t)*dt
        """
        for ball in self.balls:
            ball.update_position(self.dt)

    def _apply_boundary_conditions(self):
        """Apply periodic or bounce boundary conditions to all balls."""
        for ball in self.balls:
            ball.apply_periodic_boundary_conditions(self.well_radius)

    def calculate_average_temperature(self):
        """
        Compute the average temperature of the system.

        Scientific Equation:
        T = (2/3) * (KE / k_B)
        """
        if not self.balls:
            return 0.0
        total_ke = sum(ball.calculate_kinetic_energy() for ball in self.balls)
        return (2.0 / 3.0) * (total_ke / (len(self.balls) * 8.31e-7))

    def calculate_total_energy(self):
        """
        Compute the total energy (kinetic + potential) of the system.
        """
        total_ke = sum(ball.calculate_kinetic_energy() for ball in self.balls)
        total_pe = 0.0
        for i in range(len(self.balls)):
            for j in range(i + 1, len(self.balls)):
                total_pe += self.balls[i].calculate_potential_energy(self.balls[j], self.interaction_params)
        return total_ke + total_pe

    def run(self, target_temperature=None, thermostat_method=None):
        """
        Run the simulation from t=0 to t=total_time.

        Args:
            target_temperature (float): Target temperature for thermostat.
            thermostat_method (str): "nose_hoover" or "langevin".
        """
        current_time = 0.0
        while current_time < self.total_time:
            self.update()
            if target_temperature and thermostat_method:
                self.apply_thermostat(target_temperature, method=thermostat_method)
            current_time += self.dt
