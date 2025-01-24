import unittest
import numpy as np
from ball_simulation.well import Well
from ball_simulation.ball import Ball

class TestWell(unittest.TestCase):

    def setUp(self):
        """ Set up a cylindrical well for testing """
        self.well = Well(radius=1.0, height=2.0)
        self.ball = Ball(mass=1.0, initial_position=[0.5, 0.5, 0.5], initial_velocity=[0, 0, 0], species="O")

    def test_well_initialization(self):
        """ Test well parameters are set correctly """
        self.assertEqual(self.well.radius, 1.0)
        self.assertEqual(self.well.height, 2.0)

    def test_apply_pbc(self):
        """ Test periodic boundary conditions handling """
        self.ball.position = np.array([1.5, 0.5, 0.5])  # Outside boundary
        self.well.apply_pbc(self.ball)
        self.assertLessEqual(self.ball.position[0], self.well.radius)

    def test_compute_wall_repulsion_force(self):
        """ Test wall repulsion force calculation """
        self.ball.position = np.array([0.9, 0.0, 0.0])  # Near boundary
        force = self.well.compute_wall_repulsion_force(self.ball)
        self.assertTrue(np.any(force != 0))

if __name__ == '__main__':
    unittest.main()
