# plotting.py

import numpy as np
import matplotlib.pyplot as plt
# from ball import Ball           # If needed
# from simulation import Simulation

class SimulationPlotter:
    def __init__(self, simulation):

        self.sim = simulation

    def plot_potential_energy(self):
        # Get data from simulation
        sim_data, analytical_data = self.sim.compute_potential_energy_data()

        if not sim_data:
            print("No distance data available.")
            return

        # Extract simulation data
        distances, potential_energies = zip(*sim_data)

        # Extract analytical data
        analytical_distances, analytical_potentials = zip(*analytical_data)

        plt.figure(figsize=(8, 6))

        # Plot the analytical Lennard-Jones curve
        plt.plot(analytical_distances, analytical_potentials, color='red', label="Analytical LJ")

        # Plot simulation data
        plt.scatter(distances, potential_energies, color="blue", alpha=0.6, label="Simulated Data")

        plt.xlabel("Distance (Å)")
        plt.ylabel("Potential Energy (eV)")
        plt.title("Potential Energy vs Distance")
        plt.legend()
        plt.grid(True)
        plt.show()

    #def plot_collective_variables(self):
        # Plot total energy
     #   plt.figure()
     #   plt.plot(self.sim.collective_variables["total_energy"], label="Total Energy")
     #   plt.xlabel("Step")
     #   plt.ylabel("Energy (?)")
     #   plt.title("Total Energy vs Step")
     #   plt.grid(True)
     #   plt.legend()
     #   plt.show()

        # Plot RDF (last entry)
     #   if self.sim.collective_variables["radial_distribution"]:
     #       bins, rdf_vals = self.sim.collective_variables["radial_distribution"][-1]
     #       plt.figure()
     #       plt.bar(bins, rdf_vals, width=(bins[1]-bins[0]), alpha=0.6)
     #       plt.xlabel("Distance (Å)")
     #       plt.ylabel("g(r)")
     #       plt.title("Radial Distribution Function")
     #       plt.grid(True)
     #       plt.show()

    def plot_all(self):
        self.plot_potential_energy()
     #   self.plot_collective_variables()