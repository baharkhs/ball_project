# plotting.py

import numpy as np
import matplotlib.pyplot as plt
# from ball import Ball           # If needed
# from simulation import Simulation

import numpy as np
import matplotlib.pyplot as plt

class SimulationPlotter:
    def __init__(self, simulation):
        self.sim = simulation

    def plot_potential_energy(self):
        """
        Plots both Lennard-Jones and Coulombic potential energy as a function of distance.
        """
        # Get data from simulation
        sim_data, analytical_data = self.sim.compute_potential_energy_data()

        if not sim_data:
            print("No distance data available.")
            return

        # Extract simulation data
        distances, potential_energies = zip(*sim_data)

        # Extract analytical Lennard-Jones data
        analytical_distances, analytical_potentials = zip(*analytical_data)

        # Define a range of distances for plotting Coulombic potential
        r_values = np.linspace(0.1, 3.0, 200)  # Distance range in angstroms

        # Compute Coulombic potential (without specific charges)
        # q1 and q2 must be defined separately in the simulation class
        epsilon_0 = 8.854187817e-12  # Vacuum permittivity in SI units
        k_e = 1 / (4 * np.pi * epsilon_0)  # Coulomb's constant

        # Placeholder values for q1 and q2 (must be set elsewhere)
        q1 = 1  # Charge of particle 1 (to be set properly)
        q2 = 1  # Charge of particle 2 (to be set properly)

        # Coulombic potential formula: V = k * (q1 * q2) / r
        coulombic_potential = k_e * (q1 * q2) / r_values

        # Plot the Lennard-Jones and Coulombic interactions
        plt.figure(figsize=(8, 6))

        # Lennard-Jones: Analytical and Simulated
        plt.plot(analytical_distances, analytical_potentials, color='red', label="Analytical LJ")
        plt.scatter(distances, potential_energies, color="blue", alpha=0.6, label="Simulated LJ")

        # Coulombic potential
        plt.plot(r_values, coulombic_potential, color="green", linestyle="dashed", label="Coulombic Interaction")

        # Labels and formatting
        plt.xlabel("Distance (Å)")
        plt.ylabel("Potential Energy (eV or arbitrary units)")
        plt.title("Potential Energy: Lennard-Jones vs. Coulombic")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_all(self):
        """ Calls all relevant plotting functions. """
        self.plot_potential_energy()


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

        # Plot RDF (last entry)<
     #   if self.sim.collective_variables["radial_distribution"]:
     #       bins, rdf_vals = self.sim.collective_variables["radial_distribution"][-1]
     #       plt.figure()
     #       plt.bar(bins, rdf_vals, width=(bins[1]-bins[0]), alpha=0.6)
     #       plt.xlabel("Distance (Å)")
     #       plt.ylabel("g(r)")
     #       plt.title("Radial Distribution Function")
     #       plt.grid(True)
     #       plt.show()
