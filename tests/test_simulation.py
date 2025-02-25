import unittest
import numpy as np
from ball_simulation.simulation import Simulation
from ball_simulation.ball import Ball
from ball_simulation.well import Well

class TestSimulation(unittest.TestCase):
    def setUp(self):
        """Set up a test Simulation instance before each test."""
        self.sim = Simulation(well_radius=2.0, well_height=2.0, total_time=1.0, dt=0.001)

    def test_init(self):
        """Test Simulation initialization with correct attributes."""
        self.assertEqual(self.sim.well.radius, 2.0)
        self.assertEqual(self.sim.well.height, 2.0)
        self.assertEqual(self.sim.dt, 0.001)
        self.assertEqual(self.sim.total_time, 1.0)
        self.assertEqual(self.sim.movement_type, "newtonian")
        self.assertEqual(len(self.sim.balls), 0)
        self.assertEqual(len(self.sim.molecules), 0)

    def test_create_water_molecule(self):
        """Test creating a water molecule with fixed geometry."""
        self.sim.create_water_molecule(O_position=[0.0, 0.0, 1.0], H1_z=1.2, H2_z=0.8,
                                      velocity=[0.1, 0.1, 0.1], molecule_id="test")
        self.assertEqual(len(self.sim.balls), 3)  # O, H1, H2
        self.assertIn("test", self.sim.molecules)
        self.assertEqual(len(self.sim.molecule_com), 1)
        # Verify initial geometry (should be 0.957 Å, 104.5°—check log output for exact values)

    def test_update(self):
        """Test simulation update with Newtonian dynamics."""
        self.sim.create_water_molecule(O_position=[0.0, 0.0, 1.0], H1_z=1.2, H2_z=0.8,
                                      velocity=[0.1, 0.1, 0.1], molecule_id="test")
        initial_pos = self.sim.molecule_com["test"]["position"].copy()
        self.sim.update()
        new_pos = self.sim.molecule_com["test"]["position"]
        self.assertFalse(np.array_equal(initial_pos, new_pos))  # Position should change
        self.assertGreater(len(self.sim.temperature_history), 0)

    def test_compute_forces(self):
        """Test force computation for intermolecular interactions and wall repulsion."""
        self.sim.create_water_molecule(O_position=[0.0, 0.0, 1.0], H1_z=1.2, H2_z=0.8,
                                      velocity=[0.1, 0.1, 0.1], molecule_id="test1")
        self.sim.create_water_molecule(O_position=[0.5, 0.5, 1.0], H1_z=1.2, H2_z=0.8,
                                      velocity=[0.1, 0.1, 0.1], molecule_id="test2")
        self.sim.compute_forces()
        for mol_id in self.sim.molecule_com:
            force = self.sim.molecule_com[mol_id]["force"]
            self.assertTrue(isinstance(force, np.ndarray))
            self.assertEqual(force.shape, (3,))

    def test_temperature(self):
        """Test temperature calculation."""
        self.sim.create_water_molecule(O_position=[0.0, 0.0, 1.0], H1_z=1.2, H2_z=0.8,
                                      velocity=[0.3, 0.3, 0.3], molecule_id="test")
        temperature = self.sim.compute_system_temperature()
        self.assertGreater(temperature, 0.0)  # Should be non-zero with non-zero velocity

if __name__ == '__main__':
    unittest.main()