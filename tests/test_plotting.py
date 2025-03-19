import unittest
import numpy as np
import matplotlib

# Use a non-interactive backend for testing
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.simulation import Simulation
from src.plotting import SimulationPlotter


class TestSimulationPlotter(unittest.TestCase):

    def setUp(self):
        """
        Create a minimal config dictionary for a simple simulation.
        This config represents a simulation with one water molecule.
        """
        self.config = {
            "well": {
                "radius": 3.0,
                "height": 5.0,
                "wall_decay_length": 0.05,
                "atom_radius": 0.1,
                "repulsion_constant": 500.0
            },
            "simulation": {
                "total_time": 1.0,
                "dt": 0.01,
                "movement_type": "newtonian",
                "initial_velocity": [0.5, 0.5, 0.5],
                "target_temperature": 300,
                "rescale_interval": 50
            },
            "particles": {
                "oxygen_molecules": [
                    {"center_position": [0.0, 0.0, 2.0], "H1_z": 2.0, "H2_z": 2.0, "molecule_id": "H2O"}
                ],
                "custom_particles": []
            },
            "interaction_params": {
                "default": {"epsilon": 0.5, "sigma": 1.0, "cutoff": 2.5, "r_min_factor": 0.8}
            },
            "ball": {
                "bond_length": 0.957,
                "half_angle_degrees": 104.5,
                "default_mass": 1.0,
                "oxygen_mass": 16.0,
                "hydrogen_mass": 1.0,
                "oxygen_color": "red",
                "hydrogen_color": "blue",
                "default_color": "gray",
                "oxygen_size": 10,
                "hydrogen_size": 6,
                "default_size": 8,
                "atom_radius": 0.1
            }
        }
        # Create a simulation instance from the config.
        self.sim = Simulation.from_config(self.config)
        # Create the plotter using the simulation.
        self.plotter = SimulationPlotter(self.sim)

    def test_init_draw(self):
        """
        Test that _init_draw() correctly sets up the plotting axes.
        """
        # Check 3D axis limits
        self.assertEqual(self.plotter.ax_sim.get_xlim(), (-self.sim.well.radius, self.sim.well.radius))
        self.assertEqual(self.plotter.ax_sim.get_ylim(), (-self.sim.well.radius, self.sim.well.radius))
        self.assertEqual(self.plotter.ax_sim.get_zlim(), (0, self.sim.well.height))

        # Check that labels and title are set
        self.assertEqual(self.plotter.ax_sim.get_xlabel(), "X (Å)")
        self.assertEqual(self.plotter.ax_sim.get_ylabel(), "Y (Å)")
        self.assertEqual(self.plotter.ax_sim.get_zlabel(), "Z (Å)")
        self.assertEqual(self.plotter.ax_sim.get_title(), "Water Molecule Simulation")

        # Check temperature axis is set up.
        self.assertEqual(self.plotter.ax_temp.get_title(), "System Temperature Over Time")
        self.assertEqual(self.plotter.ax_temp.get_xlabel(), "Step")
        self.assertEqual(self.plotter.ax_temp.get_ylabel(), "Temperature (K)")

    def test_draw_frame(self):
        """
        Test that _draw_frame() updates the simulation and returns the temperature line.

        This also indirectly tests that simulation.update() is being called.
        """
        # Call _draw_frame for frame 0.
        returned = self.plotter._draw_frame(0)
        # _draw_frame should return a tuple containing the temperature line artist.
        self.assertIsInstance(returned, tuple)
        # Ensure that the temperature history is updated.
        self.assertGreater(len(self.sim.temperature_history), 0)
        # Check that simulation has at least one ball.
        self.assertGreater(len(self.sim.balls), 0)

        # Also, check that positions is a 2D array.
        positions = np.array([ball.position for ball in self.sim.balls])
        # For a single water molecule (3 balls), positions should have shape (3,3)
        self.assertEqual(positions.shape, (3, 3))

    def test_animate_simulation_runs(self):
        """
        Test that animate_simulation() runs without errors.

        We don't actually show the plot in tests; we just ensure that it can create an animation object.
        """
        try:
            self.plotter.animate_simulation()
        except Exception as e:
            self.fail(f"animate_simulation() raised an exception: {e}")


if __name__ == '__main__':
    unittest.main()
