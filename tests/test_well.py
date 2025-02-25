import unittest
import numpy as np
from ball_simulation.well import Well
from ball_simulation.ball import Ball

class TestWell(unittest.TestCase):
    def setUp(self):
        """Set up a test Well instance before each test."""
        self.well = Well(radius=2.0, height=2.0, wall_decay_length=0.05)

    def test_init(self):
        """Test Well initialization with correct attributes."""
        self.assertEqual(self.well.radius, 2.0)
        self.assertEqual(self.well.height, 2.0)
        self.assertEqual(self.well.wall_decay_length, 0.05)

    def test_apply_pbc(self):
        """Test periodic boundary conditions in z-direction."""
        ball = Ball(initial_position=[0.0, 0.0, 2.1])  # Above height
        self.well.apply_pbc(ball)
        self.assertAlmostEqual(ball.position[2], 0.1, places=6)  # Should wrap to 0.1

        ball = Ball(initial_position=[0.0, 0.0, -0.1])  # Below 0
        self.well.apply_pbc(ball)
        self.assertAlmostEqual(ball.position[2], 1.9, places=6)  # Should wrap to 1.9

    def test_apply_pbc_com(self):
        """Test PBC and boundary enforcement for COM."""
        position = np.array([0.0, 0.0, 2.1])  # Above height
        self.well.apply_pbc_com(position, self.well.height, self.well.radius)
        self.assertAlmostEqual(position[2], 0.1, places=6)  # Z wraps

        position = np.array([2.2, 0.0, 0.0])  # Beyond radius
        self.well.apply_pbc_com(position, self.well.height, self.well.radius)
        r_xy = np.linalg.norm(position[:2])
        self.assertAlmostEqual(r_xy, 1.9, places=5)  # X-Y capped at effective radius (2.0 - 0.1), allow slight precision

    def test_compute_wall_repulsion_force(self):
        """Test wall repulsion force for an atom."""
        ball = Ball(initial_position=[1.9, 0.0, 0.0])  # Near wall (r_xy = 1.9, effective radius = 1.9)
        force = self.well.compute_wall_repulsion_force(ball)
        self.assertTrue(isinstance(force, np.ndarray))
        self.assertEqual(force.shape, (3,))
        self.assertLess(force[0], -1e-6)  # Should repel inward (negative x, small threshold for precision)

        ball = Ball(initial_position=[0.0, 0.0, 0.0])  # Inside, no repulsion
        force = self.well.compute_wall_repulsion_force(ball)
        np.testing.assert_array_almost_equal(force, np.zeros(3), decimal=6)

    def test_wall_repulsion_threshold(self):
        """Test wall repulsion starts at the correct threshold (effective radius - wall_decay_length)."""
        ball = Ball(initial_position=[1.84, 0.0, 0.0])  # Just inside effective radius - wall_decay_length (1.9 - 0.05 = 1.85)
        force = self.well.compute_wall_repulsion_force(ball)
        np.testing.assert_array_almost_equal(force, np.zeros(3), decimal=6)  # No repulsion yet

        ball = Ball(initial_position=[1.86, 0.0, 0.0])  # Slightly beyond threshold
        force = self.well.compute_wall_repulsion_force(ball)
        self.assertTrue(isinstance(force, np.ndarray))
        self.assertEqual(force.shape, (3,))
        self.assertLess(force[0], -1e-6)  # Should repel inward

if __name__ == '__main__':
    unittest.main()