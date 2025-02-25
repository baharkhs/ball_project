import matplotlib.pyplot as plt
from ball_simulation.simulation import Simulation
from ball_simulation.plotting import SimulationPlotter

# Create a simulation instance (larger well for visibility).
sim = Simulation(well_radius=5.0, well_height=2.0, total_time=5.0, dt=0.001)

# Define a reasonable initial velocity for 300 K.
initial_velocity = [0.3, 0.3, 0.3]

# Create one water molecule with custom z-positions for O, H1, and H2.
sim.create_water_molecule(
    O_position=[0.0, 0.0, 1.5],  # Oxygen at z = 1.5
    H1_z=1.7,  # H1 at z = 1.7 (higher than O)
    H2_z=1.3,  # H2 at z = 1.3 (lower than O)
    velocity=initial_velocity,
    molecule_id="H2O"
)

# Create a second water molecule with custom z-positions.
sim.create_water_molecule(
    O_position=[0.0, 0.0, 0.5],  # Oxygen at z = 0.5
    H1_z=0.6,  # H1 at z = 0.7
    H2_z=0.3,  # H2 at z = 0.3
    velocity=initial_velocity,
    molecule_id="H2O2"
)

# Instantiate the plotter and run the animation.
plotter = SimulationPlotter(sim)
plotter.animate_simulation()