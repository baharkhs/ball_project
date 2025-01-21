# plotting.py

import numpy as np
import matplotlib.pyplot as plt
# from ball import Ball           # If needed
# from simulation import Simulation

class SimulationPlotter:
    def __init__(self, simulation):
        """
        Keeps a reference to the Simulation object.
        """
        self.sim = simulation

    def plot_potential_energy(self):
        """
        Plot potential energy vs. distance for *all particle pairs* in the simulation.
        """
        distances, potential_energies = self.sim.compute_potential_energy_data()

        # If no balls, pick defaults:
        if self.sim.balls:
            pair_key = tuple(sorted([self.sim.balls[0].species, self.sim.balls[1].species]))
            params = self.sim.interaction_params.get(pair_key, {"epsilon": 1.0, "sigma": 1.0})
            epsilon, sigma = params["epsilon"], params["sigma"]
        else:
            epsilon, sigma = 1.0, 1.0

        # Theoretical LJ curve
        r_theoretical = np.linspace(0.5, 2.5, 100)
        theoretical_potential = [
            Ball.lennard_jones_potential(r, epsilon, sigma) for r in r_theoretical
        ]

        plt.figure(figsize=(10, 6))
        # Plot theoretical LJ
        plt.plot(r_theoretical, theoretical_potential, color='red',
                 label=f"Theoretical LJ (ε={epsilon}, σ={sigma})")

        # Plot data
        plt.scatter(distances, potential_energies, color="blue", alpha=0.6,
                    label="Simulation Data")

        plt.xlabel("Distance (Å)")
        plt.ylabel("Potential Energy (eV)")
        plt.title("Potential Energy vs. Distance")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_collective_variables(self):
        """
        Plots total energy over time and radial distribution function (RDF).
        """
        # 1) Plot total energy
        plt.figure(figsize=(10, 6))
        plt.plot(self.sim.collective_variables["total_energy"], label="Total Energy")
        plt.xlabel("Simulation Step")
        plt.ylabel("Energy (eV)")
        plt.title("Total Energy Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

        # 2) Plot RDF (the last stored one)
        rdf_data = self.sim.collective_variables["radial_distribution"]
        if rdf_data:
            bins, rdf = rdf_data[-1]
            plt.figure(figsize=(10, 6))
            plt.bar(bins, rdf, width=(bins[1] - bins[0]), alpha=0.6,
                    label="Radial Distribution Function")
            plt.xlabel("Distance (Å)")
            plt.ylabel("g(r)")
            plt.title("Radial Distribution Function")
            plt.legend()
            plt.grid(True)
            plt.show()

    def plot_all(self):
        """
        Calls both potential energy plot and collective variables.
        """
        self.plot_potential_energy()
        self.plot_collective_variables()
