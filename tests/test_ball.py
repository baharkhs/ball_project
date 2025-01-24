import unittest
import numpy as np
from ball_simulation.ball import Ball

class TestBall(unittest.TestCase):

    def setUp(self):
        """ Create a default ball instance for testing """
        self.ball = Ball(mass=1.0, initial_position=[0, 0, 0], initial_velocity=[1, 1, 1], species="O")

    def test_ball_initialization(self):
        """ Test if ball initializes with correct values """
        self.assertEqual(self.ball.mass, 1.0)
        self.assertTrue(np.array_equal(self.ball.position, np.array([0, 0, 0])))
        self.assertTrue(np.array_equal(self.ball.velocity, np.array([1, 1, 1])))

    def test_position_update(self):
        """ Test ball movement update """
        self.ball.update_path()
        self.assertTrue(len(self.ball.path) > 0)

    def test_force_application(self):
        """ Test force application on ball """
        initial_velocity = self.ball.velocity.copy()
        self.ball.force = np.array([0.5, 0.5, 0.5])
        acceleration = self.ball.force / self.ball.mass
        self.ball.velocity += acceleration * 0.1  # Apply force for small time step
        self.assertFalse(np.array_equal(initial_velocity, self.ball.velocity))

if __name__ == '__main__':
    unittest.main()
