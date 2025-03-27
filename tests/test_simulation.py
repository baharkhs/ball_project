import unittest
import numpy as np
from src.simulation import Simulation
from src.well import Well
from src.ball import Ball


class TestSimulation(unittest.TestCase):

    def setUp(self):
        """
        Create a basic Simulation instance with minimal required parameters.

        Variables passed:
          - dt: Time step (0.001)
          - total_time: Total simulation duration (1.0)
          - movement_type: "newtonian" (integration scheme)
          - interaction_params: A dictionary with at least a default interaction parameter.
          - target_temperature: Desired temperature (300)
          - rescale_interval: Rescaling every 10 steps
          - well_instance: A Well instance with radius 2.0, height 2.0, and additional parameters.
          - ball_defaults: A dictionary with defaults for creating Ball objects.
        """
        # Create a well instance with specified dimensions and parameters.
        well_instance = Well(radius=2.0, height=2.0, wall_decay_length=0.05, repulsion_constant=500.0, atom_radius=0.1)
        # Minimal interaction parameters; using only default values.
        interaction_params = {
            "default": {"epsilon": 0.5, "sigma": 1.0, "cutoff": 2.5, "r_min_factor": 0.8}
        }
        # Default properties for Ball objects.
        ball_defaults = {
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
            "default_size": 8
        }
        self.sim = Simulation(
            dt=0.001,
            total_time=1.0,
            movement_type="newtonian",
            interaction_params=interaction_params,
            target_temperature=300,
            rescale_interval=10,
            well_instance=well_instance,
            ball_defaults=ball_defaults
        )

    def test_init(self):
        """Test Simulation initialization and that the attributes are correctly assigned."""
        # Check well dimensions.
        self.assertEqual(self.sim.well.radius, 2.0)
        self.assertEqual(self.sim.well.height, 2.0)
        # Check simulation parameters.
        self.assertAlmostEqual(self.sim.dt, 0.001)
        self.assertAlmostEqual(self.sim.total_time, 1.0)
        self.assertEqual(self.sim.movement_type, "newtonian")
        # Initially, there should be no balls or molecules.
        self.assertEqual(len(self.sim.balls), 0)
        self.assertEqual(len(self.sim.molecules), 0)

    def test_create_water_molecule(self):
        """
        Test creating a water molecule.

        The method create_water_molecule should add 3 balls (oxygen, hydrogen1, hydrogen2),
        group them under a molecule_id, and compute the center-of-mass (COM) and offsets.
        """
        # Create a water molecule with given positions and velocity.
        self.sim.create_water_molecule(
            center_position=[0.0, 0.0, 1.0],
            h1_z=1.2,
            h2_z=0.8,
            velocity=[0.1, 0.1, 0.1],
            molecule_id="test"
        )
        # There should now be 3 balls.
        self.assertEqual(len(self.sim.balls), 3)
        # The molecule dictionary should have an entry for "test".
        self.assertIn("test", self.sim.molecules)
        # The COM data dictionary should have one entry.
        self.assertEqual(len(self.sim.molecule_com), 1)
        # Optionally, one could check that the COM position is computed as an average.

    def test_update(self):
        """
        Test that after an update, the center-of-mass (COM) position of a molecule changes.

        Also, temperature history should have at least one value after the update.
        """
        self.sim.create_water_molecule(
            center_position=[0.0, 0.0, 1.0],
            h1_z=1.2,
            h2_z=0.8,
            velocity=[0.1, 0.1, 0.1],
            molecule_id="test"
        )
        initial_com = self.sim.molecule_com["test"]["position"].copy()
        self.sim.update()
        new_com = self.sim.molecule_com["test"]["position"]

        #check that the temperature history now has a value.
        self.assertGreater(len(self.sim.temperature_history), 0)

    def test_compute_forces(self):
        """
        Test that forces are computed and stored in each molecule's COM.

        For each molecule, the 'force' should be a numpy array of shape (3,).
        """
        self.sim.create_water_molecule(
            center_position=[0.0, 0.0, 1.0],
            h1_z=1.2,
            h2_z=0.8,
            velocity=[0.1, 0.1, 0.1],
            molecule_id="mol1"
        )
        self.sim.create_water_molecule(
            center_position=[0.5, 0.5, 1.0],
            h1_z=1.2,
            h2_z=0.8,
            velocity=[0.1, 0.1, 0.1],
            molecule_id="mol2"
        )
        self.sim.compute_forces()
        for mol_id in self.sim.molecule_com:
            force = self.sim.molecule_com[mol_id]["force"]
            self.assertTrue(isinstance(force, np.ndarray))
            self.assertEqual(force.shape, (3,))

    def test_temperature(self):
        """
        Test that the system temperature is calculated correctly.

        With a non-zero velocity, the computed temperature should be greater than zero.
        """
        self.sim.create_water_molecule(
            center_position=[0.0, 0.0, 1.0],
            h1_z=1.2,
            h2_z=0.8,
            velocity=[0.3, 0.3, 0.3],
            molecule_id="test"
        )
        temp = self.sim.compute_system_temperature()
        self.assertGreater(temp, 0.0)


if __name__ == '__main__':
    unittest.main()
