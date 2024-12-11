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
    def __init__(self, position, velocity, mass, radius, attraction_coeff=0.0, repulsion_coeff=0.0):
        """
        Initialize a Ball object.

        Args:
            position (array-like): Initial position of the ball [x, y, z].
            velocity (array-like): Initial velocity of the ball [vx, vy, vz].
            mass (float): Mass of the ball.
            radius (float): Radius of the ball.
            attraction_coeff (float): Coefficient for attraction forces.
            repulsion_coeff (float): Coefficient for repulsion forces.
        """
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass
        self.radius = radius
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

    def update_position(self, dt):
        """
        Update the position and velocity of the ball using Newton's second law.

        Args:
            dt (float): Time step for the update.
        """
        acceleration = self.force / self.mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

    def apply_periodic_boundary_conditions(self, box_size):
        """
        Apply periodic boundary conditions to the ball's position.

        Args:
            box_size (float): Size of the cubic simulation box.
        """
        self.position = np.mod(self.position + box_size, box_size)

    def update_path(self, finalize=False):
        """
        Update the particle's path by recording its current position or finalize the current segment.

        Args:
            finalize (bool): If True, finalize the current path segment and start a new one.
        """
        if finalize:
            # Finalize the current segment if it has recorded points
            if self.current_path_segment["x"]:
                self.path_segments.append(self.current_path_segment)
            self.current_path_segment = {"x": [], "y": [], "z": []}  # Start a new segment
        else:
            # Append current position to the ongoing segment
            self.current_path_segment["x"].append(self.position[0])
            self.current_path_segment["y"].append(self.position[1])
            self.current_path_segment["z"].append(self.position[2])

    def get_all_paths(self):
        """
        Return all recorded path segments for visualization, including the current segment.

        Returns:
            list: A list of all path segments, finalized and ongoing.
        """
        # Include the current path segment even if not finalized
        return self.path_segments + [self.current_path_segment] if self.current_path_segment[
            "x"] else self.path_segments


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

        Internal Details:
        - Positions in Å, velocities in Å/fs, time in fs, mass in amu.
        - Forces computed in amu·Å/fs², energies in amu·(Å/fs)².
        - Temperature in Kelvin (K).

        The well defines the spatial constraints along x–y (cylindrical boundary) and z-axis (with PBC and/or repulsion).
        The `movement_type` influences how positions are updated:
        - "newtonian": Uses classical equations of motion (F=ma).
        - "monte_carlo": Introduces random steps in position and stochastic forces.
        """
        self.well = Well(radius=well_radius, height=well_height)
        self.dt = dt
        self.total_time = total_time
        self.balls = []
        self.movement_type = movement_type
        self.average_temperature = 0.0
        self.total_energy = 0.0

        # Interaction parameters for Lennard-Jones potential between species.
        # epsilon in amu*(Å/fs)², sigma in Å, cutoff in Å.
        # Adjust these parameters as needed for realism.
        self.interaction_params = {
            ("O", "O"): {"epsilon": 0.5, "sigma": 0.3, "cutoff": 5.0},
            ("H", "O"): {"epsilon": 0.3, "sigma": 0.25, "cutoff": 4.0},
            ("H", "H"): {"epsilon": 0.2, "sigma": 0.2, "cutoff": 3.0},
        }

    def set_movement_type(self, movement_type="newtonian"):
        """
        Sets the movement type for the simulation.

        Args:
            movement_type (str): "newtonian" or "monte_carlo".

        Newtonian: Deterministic update from forces.
        Monte Carlo: Random steps and force perturbations, mimicking stochastic processes.
        """
        self.movement_type = movement_type

    def add_ball(self, mass=1.0, initial_position=None, initial_velocity=None, species="O", molecule_id=None):
        """
        Adds a ball (particle) to the simulation with specified properties.

        Args:
            mass (float): Mass in amu.
            initial_position (array-like): [x, y, z] in Å.
            initial_velocity (array-like): [vx, vy, vz] in Å/fs.
            species (str): Particle type, affects LJ params.
            molecule_id (int): For identifying molecules if needed.

        The added ball is appended to self.balls.
        """
        ball = Ball(
            mass=mass,
            initial_position=initial_position,
            initial_velocity=initial_velocity,
            species=species,
            molecule_id=molecule_id,
        )
        self.balls.append(ball)

    def apply_monte_carlo_movement(self, ball):
        """
        Applies a random displacement to the ball’s position (Monte Carlo step).

        Args:
            ball (Ball): The particle to update.

        This is a simplistic random walk:
        - max_step_size in Å.
        - Chosen uniformly from [-max_step_size, max_step_size].
        """
        max_step_size = 0.01
        random_step = np.random.uniform(-max_step_size, max_step_size, size=3)
        ball.position += random_step

    def apply_monte_carlo_perturbation(self, ball):
        """
        Adds random noise to the ball’s force, simulating stochastic thermal agitation.

        Args:
            ball (Ball): The particle to which noise is applied.

        Noise magnitude: 0.1 (in amu·Å/fs²). Adjust as needed for realism.
        """
        noise_magnitude = 0.1
        noise = np.random.uniform(-noise_magnitude, noise_magnitude, size=3)
        ball.force += noise

    def rescale_temperatures(self, target_temp, decay_factor=1):
        """
        Adjusts the velocities of all balls to bring system temperature towards target_temp.
        Uses a decay factor to avoid sudden jumps.

        Args:
            target_temp (float): Target temperature in K.
            decay_factor (float): Controls smoothness of rescaling.
        """
        current_temp = self.calculate_average_temperature()
        if current_temp <= 0:
            return  # Avoid division by zero or negative temperature issues

        # Calculate the scaling factor to match the target temperature
        scaling_factor = np.sqrt(target_temp / current_temp)

        # Smoothly approach the scaling factor using the decay factor
        adjusted_factor = 1.0 - decay_factor + (decay_factor * scaling_factor)

        # Rescale velocities
        for ball in self.balls:
            ball.velocity *= adjusted_factor

    def update(self):
        """
        Performs a single simulation step:
        1. Reset forces and calculate average temperature.
        2. Compute pairwise interaction forces (LJ).
        3. Compute wall repulsion forces.
        4. Apply Monte Carlo perturbations if needed.
        5. Update velocities and positions.
        6. Apply boundary conditions and record paths.
        7. Compute total energy.

        Each call to update() advances the system by one timestep (dt).
        """
        # Step 1: Reset forces and compute average temperature
        self._reset_forces()
        self.average_temperature = self.calculate_average_temperature()

        # Step 2: Compute pairwise forces (Lennard-Jones)
        self._compute_pairwise_forces()

        # Step 3: Compute wall repulsion forces
        self._apply_wall_forces()

        # Step 4: Apply MC perturbation if movement_type is "monte_carlo"
        # (Already integrated in _apply_wall_forces for convenience, but can be separate if desired.)

        # Step 5: Update velocities and positions according to movement type
        self._update_positions_velocities()

        # Step 6: Apply boundary conditions (bounce and PBC) and record paths
        self._apply_boundary_conditions_and_update_paths()

        # Step 7: Compute total energy for bookkeeping
        self.total_energy = self.calculate_total_energy()

        # Removed debug prints of forces and velocities to avoid clutter and confusion.
        # If desired, you can uncomment the following line or add conditional logging:
        # for ball in self.balls:
        #     print(f"Ball {ball.species}: Force = {ball.force}, Velocity = {ball.velocity}")

    def _reset_forces(self):
        """
        Reset forces on all balls before recalculating them.
        """
        for ball in self.balls:
            ball.force = np.zeros(3)

    def _compute_pairwise_forces(self):
        """
        Compute and apply Lennard-Jones interaction forces between all pairs of balls.
        """
        num_balls = len(self.balls)
        for i in range(num_balls):
            for j in range(i + 1, num_balls):
                # Compute force on ball i due to ball j
                interaction_force = self.balls[i].compute_interaction_force(self.balls[j], self.interaction_params)
                # A small safeguard to ensure no NaN forces:
                if np.any(np.isnan(interaction_force)):
                    interaction_force = np.zeros(3)
                self.balls[i].force += interaction_force
                self.balls[j].force -= interaction_force  # Newton's Third Law: equal and opposite force

    def _apply_wall_forces(self):
        """
        Compute wall repulsion forces for each ball and add them to their net force.
        If monte_carlo dynamics is selected, also apply random perturbation to forces.
        """
        for ball in self.balls:
            wall_force = self.well.compute_wall_repulsion_force(ball)
            # Check for NaN again just for safety
            if np.any(np.isnan(wall_force)):
                wall_force = np.zeros(3)
            ball.force += wall_force
            if self.movement_type == "monte_carlo":
                # Add random noise to simulate stochastic environment
                self.apply_monte_carlo_perturbation(ball)

    def _update_positions_velocities(self):
        """
        Update the positions and velocities of balls:
        - For "monte_carlo", apply random steps to position.
        - For "newtonian", use ball.update_velocity_position(dt) to integrate equations of motion.
        """
        for ball in self.balls:
            if self.movement_type == "monte_carlo":
                # Ignore forces for position update; purely random step
                self.apply_monte_carlo_movement(ball)
            else:
                # Newtonian dynamics: v(t+dt) = v(t) + (F/m)*dt; x(t+dt) = x(t) + v(t+dt)*dt
                ball.update_velocity_position(self.dt)

    def _apply_boundary_conditions_and_update_paths(self):
        """
        Apply boundary conditions (bounce and PBC) and update path information.
        """
        for ball in self.balls:
            # Apply bounce in x-y if needed (non-physical but retained)
            ball.velocity = self.well.apply_bounce(ball)

            # Apply periodic boundary conditions along z if configured
            new_position, wrapped = self.well.apply_pbc(ball.position)
            ball.position = new_position

            # If we wrapped the particle, skip this point in the path to avoid visual jumps
            ball.skip_path_update = True if wrapped else False
            ball.update_path()

    def calculate_average_temperature(self):
        """
        Compute the average temperature of all balls.

        Returns:
            float: Average T in K.
        """
        if not self.balls:
            return 0.0
        temperatures = [ball.calculate_temperature() for ball in self.balls]
        return sum(temperatures) / len(temperatures)

    def calculate_total_energy(self):
        """
        Compute the total (kinetic + potential) energy of the system.

        Returns:
            float: Total energy in amu*(Å/fs)².
        """
        total_kinetic = sum(ball.calculate_kinetic_energy() for ball in self.balls)
        # Potential energy: sum over pairs
        total_potential = sum(
            self.balls[i].calculate_potential_energy(self.balls[j], self.interaction_params)
            for i in range(len(self.balls)) for j in range(i + 1, len(self.balls))
        )
        return total_kinetic + total_potential

    def finalize_simulation(self):
        """
        Finalize simulation:
        - Ensure all path segments are recorded.
        - Any cleanup required at the end of simulation.

        This is a placeholder for additional end-of-run tasks.
        """
        for ball in self.balls:
            ball.finalize_path()

    def run(self, target_temperature=None):
        """
        Run the simulation from t=0 to t=total_time, updating at each dt.

        Args:
            target_temperature (float): If provided, periodically rescale velocities to achieve this temperature.

        The simulation progresses in discrete steps:
        - Calls update() each time step.
        - Optionally rescales velocities at intervals to maintain target temperature.
        - Prints debugging info (temperatures).

        Consider more sophisticated thermostats (e.g., Langevin, Berendsen, Nosé-Hoover) for realistic temperature control.
        """
        current_time = 0.0
        rescale_interval = 10  # Every 10 steps, attempt to rescale if target_temperature is set
        step_count = 0

        while current_time < self.total_time:
            self.update()

            # If a target temperature is specified, rescale velocities periodically
            if target_temperature is not None and step_count % rescale_interval == 0:
                print(f"Rescaling velocities at step {step_count} to target temperature {target_temperature} K")
                self.rescale_temperatures(target_temperature)

            # Debug: Print average temperature at each step
            #print(f"Step {step_count}: Average Temperature = {self.average_temperature:.2f} K")

            current_time += self.dt
            step_count += 1

        # After completing the simulation time, finalize and store paths
        self.finalize_simulation()




