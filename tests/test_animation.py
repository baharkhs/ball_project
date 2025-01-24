import unittest
import numpy as np
from unittest.mock import patch
from ball_simulation.simulation import Simulation

class TestAnimation(unittest.TestCase):
    @patch('ball_simulation.simulation.np.random.normal')
    def test_update_function(self, mock_random):
        """
        Test the update function to ensure that it changes the positions of the balls
        when the random normal movement is applied, simulating a Monte Carlo step.
        """
        # Mock random movement to control test behavior
        mock_random.return_value = np.array([0.1, 0.1, 0.1])  # Simulated movement perturbation

        # Initialize the simulation with parameters
        sim = Simulation(well_radius=0.6, well_height=1.0, total_time=15.0, dt=0.02)
        sim.add_ball(mass=2.0, initial_position=[0.2, 0.6, 0.5], initial_velocity=[0.1, -0.05, 0.05])
        sim.add_ball(mass=1.2, initial_position=[-0.3, 0.4, 0.3], initial_velocity=[0, 0, 0])

        # Capture initial positions
        initial_positions = [ball.position.copy() for ball in sim.balls]

        # Run the update function to simulate movement
        sim.update()

        # Assert that at least one ball's position has changed
        for i, ball in enumerate(sim.balls):
            self.assertFalse(np.array_equal(ball.position, initial_positions[i]), f"Ball {i + 1} position did not change.")

if __name__ == '__main__':
    unittest.main()
