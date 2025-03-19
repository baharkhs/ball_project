import unittest
import numpy as np
from src.well import Well
from src.ball import Ball


class TestWell(unittest.TestCase):
    def setUp(self):
        """
        Set up a test Well instance before each test.

        Parameters chosen:
          - radius: 2.0 (the radius of the cylindrical container)
          - height: 2.0 (the height of the container)
          - wall_decay_length: 0.05 (how quickly the repulsive force decays)
          - repulsion_constant: 500.0 (strength of the repulsive force)
          - atom_radius: 0.1 (radius of a particle)
        """
        self.well = Well(radius=2.0, height=2.0, wall_decay_length=0.05, repulsion_constant=500.0, atom_radius=0.1)

    def test_init(self):
        """Test that the Well is initialized with the correct attributes."""
        self.assertEqual(self.well.radius, 2.0)
        self.assertEqual(self.well.height, 2.0)
        self.assertEqual(self.well.wall_decay_length, 0.05)
        self.assertEqual(self.well.repulsion_constant, 500.0)
        self.assertEqual(self.well.atom_radius, 0.1)

    def test_apply_pbc(self):
        """
        Test the periodic boundary conditions (PBC) applied by apply_pbc().

        For the z-coordinate:
          - If a ball is above the height (e.g. 2.1), it should wrap around (2.1 mod 2.0 = 0.1).
          - If below 0 (e.g. -0.1), it should wrap to 1.9.
        For the x-y plane:
          - If a ball is outside the circle (e.g. at (3.0, 0.0)), its x-y coordinates are scaled to lie on the circle.
        """
        # Create a ball at z = 2.1 (above the well's height)
        ball = Ball("X", [0.0, 0.0, 2.1], [0, 0, 0])
        self.well.apply_pbc(ball)
        self.assertAlmostEqual(ball.position[2], 0.1, places=6)

        # Create a ball at z = -0.1 (below 0)
        ball = Ball("X", [0.0, 0.0, -0.1], [0, 0, 0])
        self.well.apply_pbc(ball)
        self.assertAlmostEqual(ball.position[2], 1.9, places=6)

        # Test x-y scaling: place a ball at (3.0, 0.0, 1.0) and expect its radial distance to be 2.0
        ball = Ball("X", [3.0, 0.0, 1.0], [0, 0, 0])
        self.well.apply_pbc(ball)
        r_xy = np.linalg.norm(ball.position[:2])
        self.assertAlmostEqual(r_xy, 2.0, places=6)

    def test_compute_wall_repulsion_force(self):
        """
        Test the wall repulsion force for a ball near the wall.

        When a ball is near the wall, it should experience a repulsive force directed inward in the x-y plane.
        The z-component should be zero because the wall is only on the sides.
        """
        # Place a ball near the wall: effective boundary = radius - atom_radius = 2.0 - 0.1 = 1.9.
        # For example, a ball at (1.9, 0.0, 1.0) is right at the edge.
        ball = Ball("X", [1.9, 0.0, 1.0], [0, 0, 0])
        force = self.well.compute_wall_repulsion_force(ball)
        self.assertTrue(isinstance(force, np.ndarray))
        self.assertEqual(force.shape, (3,))
        # Expect a repulsive force along the x-axis that is negative (pushing inward)
        self.assertLess(force[0], -1e-6)
        # The z-component should be 0.
        self.assertAlmostEqual(force[2], 0.0, places=6)

        # Test a ball well inside the boundary: no repulsive force should be applied.
        ball = Ball("X", [0.0, 0.0, 1.0], [0, 0, 0])
        force = self.well.compute_wall_repulsion_force(ball)
        np.testing.assert_array_almost_equal(force, np.zeros(3), decimal=6)

    def test_wall_repulsion_threshold(self):
        """
        Test that wall repulsion starts only when the ball crosses the threshold.

        The effective boundary is (radius - atom_radius), which is 1.9 in this case.
        The threshold for repulsion is defined as (effective boundary - wall_decay_length) = 1.9 - 0.05 = 1.85.
        A ball with radial distance less than 1.85 should experience no repulsion.
        """
        # Ball just inside the threshold (r = 1.84 < 1.85): no repulsion.
        ball = Ball("X", [1.84, 0.0, 1.0], [0, 0, 0])
        force = self.well.compute_wall_repulsion_force(ball)
        np.testing.assert_array_almost_equal(force, np.zeros(3), decimal=6)

        # Ball just beyond the threshold (r = 1.86 > 1.85): should have repulsion.
        ball = Ball("X", [1.86, 0.0, 1.0], [0, 0, 0])
        force = self.well.compute_wall_repulsion_force(ball)
        self.assertTrue(isinstance(force, np.ndarray))
        self.assertEqual(force.shape, (3,))
        self.assertLess(force[0], -1e-6)  # Negative value indicates inward repulsion


if __name__ == '__main__':
    unittest.main()
