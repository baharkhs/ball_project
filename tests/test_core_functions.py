import unittest
from ball_simulation.core_functions import CoreFunctions

class TestCoreFunctions(unittest.TestCase):
    def setUp(self):
        # Set up initial conditions for the ball
        self.ball = CoreFunctions(position=0, velocity=0, acceleration=9.81)  # Example: free fall

    def test_update_position(self):
        # Test position update over a time step
        time_step = 1.0  # 1 second
        new_position = self.ball.update_position(time_step)
        self.assertAlmostEqual(new_position, 4.905, places=2)  # Position after 1 second under gravity

if __name__ == '__main__':
    unittest.main()
