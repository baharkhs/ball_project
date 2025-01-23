# plotting.py

import numpy as np
import matplotlib.pyplot as plt
# from ball import Ball           # If needed
# from simulation import Simulation

class SimulationPlotter:
    def __init__(self, simulation):

        self.sim = simulation

    def plot_potential_energy(self):
        # Use self.sim.compute_potential_energy_data() to get distances & energies
        distances, potential_energies = self.sim.compute_potential_energy_data()

        if not distances:
            print("No distance data available.")
            return

        # Optionally, if you want a theoretical LJ curve:
        if len(self.sim.balls) > 1:
            pair_key = tuple(sorted([self.sim.balls[0].species, self.sim.balls[1].species]))
            params = self.sim.interaction_params.get(pair_key, {"epsilon":1.0,"sigma":1.0})
            epsilon, sigma = params["epsilon"], params["sigma"]
        else:
            epsilon, sigma = 1.0, 1.0

        # Theoretical curve
        r_vals = np.linspace(0.5, 2.5, 100)
        pot_theory = [Ball.lennard_jones_potential(r, epsilon, sigma) for r in r_vals]

        plt.figure(figsize=(8,6))
        plt.plot(r_vals, pot_theory, color='red', label=f"LJ(e={epsilon}, s={sigma})")
        plt.scatter(distances, potential_energies, color="blue", alpha=0.6, label="Sim data")
        plt.xlabel("Distance (Å)")
        plt.ylabel("Potential (eV)")
        plt.title("Potential Energy vs Distance")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_collective_variables(self):
        # Plot total energy
        plt.figure()
        plt.plot(self.sim.collective_variables["total_energy"], label="Total Energy")
        plt.xlabel("Step")
        plt.ylabel("Energy (?)")
        plt.title("Total Energy vs Step")
        plt.grid(True)
        plt.legend()
        plt.show()

        # Plot RDF (last entry)
        if self.sim.collective_variables["radial_distribution"]:
            bins, rdf_vals = self.sim.collective_variables["radial_distribution"][-1]
            plt.figure()
            plt.bar(bins, rdf_vals, width=(bins[1]-bins[0]), alpha=0.6)
            plt.xlabel("Distance (Å)")
            plt.ylabel("g(r)")
            plt.title("Radial Distribution Function")
            plt.grid(True)
            plt.show()

    def plot_all(self):
        self.plot_potential_energy()
        self.plot_collective_variables()