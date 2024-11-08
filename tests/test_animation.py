import unittest
import numpy as np
from unittest.mock import patch
from ball_simulation.animation import Simulation  # Import your main Simulation class

class TestAnimation(unittest.TestCase):
    @patch('numpy.random.normal')
    def test_update_function(self, mock_random):
        """
        Test the update function to ensure that it changes the positions of the balls
        when the random normal movement is applied, simulating a Monte Carlo step.
        """
        # Set the return value of the mock to ensure a specific change in random movement
        mock_random.return_value = np.array([0.1, 0.1, 0.1])  # Force a specific change in position

        # Initialize the simulation with a sample well and two balls
        sim = Simulation(well_radius=0.6, well_height=1.0, total_time=15.0, dt=0.02)
        sim.add_ball(mass=2.0, initial_position=[0.2, 0.6, 0.5], initial_velocity=[0.1, -0.05, 0.05], movement_type="newtonian")
        sim.add_ball(mass=1.2, initial_position=[-0.3, 0.4, 0.3], initial_velocity=[0, 0, 0], movement_type="monte_carlo")

        # Save initial positions of both balls
        initial_positions = [ball.position.copy() for ball in sim.balls]

        # Call the update function for the simulation
        sim.update()  # This should apply the random changes to position if Monte Carlo

        # Test that positions of both balls have changed after the update
        for i, ball in enumerate(sim.balls):
            self.assertFalse(np.array_equal(ball.position, initial_positions[i]), f"Ball {i + 1} position did not change.")

if __name__ == '__main__':
    unittest.main()
