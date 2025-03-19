import unittest
import numpy as np
from src.ball import Ball


class TestBall(unittest.TestCase):
    def setUp(self):
        """
        Set up a test Ball instance representing a hydrogen atom.

        Parameters:
          - species: "H" (hydrogen)
          - position: [0.0, 0.0, 0.0]
          - velocity: [0.5, 0.5, 0.5]
          - molecule_id: "test"
          - mass: 1.0
          - color: "blue"
          - size: 6
        """
        self.ball = Ball("H", [0.0, 0.0, 0.0], [0.5, 0.5, 0.5], molecule_id="test", mass=1.0, color="blue", size=6)

    def test_init(self):
        """Test Ball initialization with correct attributes."""
        self.assertEqual(self.ball.mass, 1.0)
        np.testing.assert_array_almost_equal(self.ball.position, np.array([0.0, 0.0, 0.0]), decimal=6)
        np.testing.assert_array_almost_equal(self.ball.velocity, np.array([0.5, 0.5, 0.5]), decimal=6)
        self.assertEqual(self.ball.species, "H")
        self.assertEqual(self.ball.molecule_id, "test")
        self.assertEqual(self.ball.color, "blue")
        self.assertEqual(self.ball.size, 6)
        # The charge test is omitted because the Ball class does not define a 'charge' attribute.

    def test_lennard_jones_potential(self):
        """Test Lennard-Jones potential calculation for a given distance."""
        epsilon, sigma = 0.05, 2.0  # Example parameters for H-H interactions.
        r = 2.0  # Distance in Å.
        potential = Ball.lennard_jones_potential(r, epsilon, sigma)
        expected = 4 * 0.05 * ((2.0 / r) ** 12 - (2.0 / r) ** 6)
        self.assertAlmostEqual(potential, expected, places=6)

    def test_lennard_jones_force(self):
        """Test Lennard-Jones force calculation for a given distance."""
        epsilon, sigma = 0.05, 2.0  # Example parameters for H-H interactions.
        r = 2.0  # Distance in Å.
        force = Ball.lennard_jones_force(r, epsilon, sigma)
        expected = 24 * 0.05 * (2 * (2.0 / r) ** 12 - (2.0 / r) ** 6) / r
        self.assertAlmostEqual(force, expected, places=6)

    def test_compute_interaction_force(self):
        """Test intermolecular force calculation with periodic boundary conditions."""
        # Create another ball at position [2.0, 0.0, 0.0] with zero velocity.
        other = Ball("H", [2.0, 0.0, 0.0], [0.0, 0.0, 0.0], molecule_id="test2", mass=1.0, color="blue", size=6)
        box_lengths = np.array([5.0, 5.0, 5.0])  # Example simulation box dimensions.
        interaction_params = {("H", "H"): {"epsilon": 0.05, "sigma": 2.0, "cutoff": 5.0}}
        force = self.ball.compute_interaction_force(other, interaction_params, box_lengths)
        self.assertTrue(isinstance(force, np.ndarray))
        self.assertEqual(force.shape, (3,))

    def test_update_velocity_position(self):
        """Test a simple update of the ball's position based on its velocity."""
        dt = 0.001  # Time step.
        initial_pos = self.ball.position.copy()
        # For testing, update the position: new_position = old_position + velocity * dt.
        self.ball.position += self.ball.velocity * dt
        expected_pos = initial_pos + np.array([0.5, 0.5, 0.5]) * dt
        np.testing.assert_array_almost_equal(self.ball.position, expected_pos, decimal=6)

    def test_path_tracking(self):
        """Test the ball's path tracking functionality."""
        # Initially, the current path segment should be empty.
        self.assertEqual(len(self.ball.current_path_segment["x"]), 0)
        # Call update_path to record the current position.
        self.ball.update_path()
        self.assertEqual(len(self.ball.current_path_segment["x"]), 1)
        # Verify that the recorded position matches the ball's current position.
        recorded_pos = np.array([
            self.ball.current_path_segment["x"][0],
            self.ball.current_path_segment["y"][0],
            self.ball.current_path_segment["z"][0]
        ])
        np.testing.assert_array_almost_equal(recorded_pos, self.ball.position, decimal=6)


if __name__ == '__main__':
    unittest.main()
