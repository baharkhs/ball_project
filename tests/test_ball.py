import unittest
import numpy as np
from ball_simulation.ball import Ball

class TestBall(unittest.TestCase):
    def setUp(self):
        """Set up a test Ball instance before each test."""
        self.ball = Ball(mass=1.0, initial_position=[0.0, 0.0, 0.0], initial_velocity=[0.5, 0.5, 0.5],
                         species="H", molecule_id="test", color="blue", size=6)

    def test_init(self):
        """Test Ball initialization with correct attributes."""
        self.assertEqual(self.ball.mass, 1.0)
        self.assertTrue(np.array_equal(self.ball.position, np.array([0.0, 0.0, 0.0])))
        self.assertTrue(np.array_equal(self.ball.velocity, np.array([0.5, 0.5, 0.5])))  # Matches default velocity
        self.assertEqual(self.ball.species, "H")
        self.assertEqual(self.ball.molecule_id, "test")
        self.assertEqual(self.ball.color, "blue")
        self.assertEqual(self.ball.size, 6)
        self.assertEqual(self.ball.radius, 0.1)
        self.assertAlmostEqual(self.ball.charge, 0.417, places=3)

    def test_lennard_jones_potential(self):
        """Test Lennard-Jones potential calculation."""
        epsilon, sigma = 0.05, 2.0  # H-H parameters
        r = 2.0  # Distance in Å
        potential = Ball.lennard_jones_potential(r, epsilon, sigma)
        expected = 4 * 0.05 * ((2.0 / r) ** 12 - (2.0 / r) ** 6)
        self.assertAlmostEqual(potential, expected, places=6)

    def test_lennard_jones_force(self):
        """Test Lennard-Jones force calculation."""
        epsilon, sigma = 0.05, 2.0  # H-H parameters
        r = 2.0  # Distance in Å
        force = Ball.lennard_jones_force(r, epsilon, sigma)
        expected = 24 * 0.05 * (2 * (2.0 / r) ** 12 - (2.0 / r) ** 6) / r
        self.assertAlmostEqual(force, expected, places=6)

    def test_compute_interaction_force(self):
        """Test intermolecular force calculation with PBC."""
        other = Ball(mass=1.0, initial_position=[2.0, 0.0, 0.0], initial_velocity=[0.0, 0.0, 0.0],
                     species="H", molecule_id="test2")
        box_lengths = np.array([5.0, 5.0, 5.0])  # Example box lengths
        interaction_params = {("H", "H"): {"epsilon": 0.05, "sigma": 2.0, "cutoff": 5.0}}
        force = self.ball.compute_interaction_force(other, interaction_params, box_lengths)
        self.assertTrue(isinstance(force, np.ndarray))
        self.assertEqual(force.shape, (3,))

    def test_update_velocity_position(self):
        """Test position and velocity updates."""
        dt = 0.001  # Time step in fs
        initial_pos = self.ball.position.copy()
        initial_vel = self.ball.velocity.copy()
        self.ball.update_velocity_position(dt)
        expected_pos = initial_pos + initial_vel * dt
        np.testing.assert_array_almost_equal(self.ball.position, expected_pos, decimal=6)

    def test_path_tracking(self):
        """Test path tracking functionality."""
        self.ball.update_path()
        self.assertEqual(len(self.ball.current_path_segment["x"]), 1)  # Now should be 1 after update
        self.assertEqual(self.ball.current_path_segment["x"][0], 0.0)
        self.assertEqual(self.ball.current_path_segment["y"][0], 0.0)
        self.assertEqual(self.ball.current_path_segment["z"][0], 0.0)

if __name__ == '__main__':
    unittest.main()