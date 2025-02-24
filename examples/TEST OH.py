import matplotlib.pyplot as plt
from ball_simulation.simulation import Simulation
from ball_simulation.plotting import SimulationPlotter

# Create a simulation instance (a well large enough for visualization).
sim = Simulation(well_radius=2.0, well_height=2.0, total_time=5.0, dt=0.001)

# Define a nonzero initial velocity.
initial_velocity = [1.2, 1.2, 2.2]

# Create one water molecule (1 O and 2 H) with molecule_id "H2O".
sim.create_water_molecule(center_position=[0.0, 0.0, 1.5],
                          velocity=initial_velocity,
                          molecule_id="H2O")

# Create a second water molecule.
sim.create_water_molecule(center_position=[0.0, 0.0, 0.5],
                          velocity=initial_velocity,
                          molecule_id="H2O2")

# Instantiate the plotter and run the animation and additional plots.
plotter = SimulationPlotter(sim)
plotter.animate_simulation()

