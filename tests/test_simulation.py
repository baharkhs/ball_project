import unittest
import numpy as np
from ball_simulation.simulation import Simulation
from ball_simulation.ball import Ball
from ball_simulation.well import Well


class TestSimulation(unittest.TestCase):

    def setUp(self):
        """ Set up a default simulation environment for testing """
        self.sim = Simulation(well_radius=0.6, well_height=1.0, total_time=5.0, dt=0.001)

    def test_add_ball(self):
        """ Test adding a ball to the simulation """
        initial_count = len(self.sim.balls)
        self.sim.add_ball(mass=1.5, initial_position=[0.2, 0.4, 0.6], initial_velocity=[0, 0, 0], species="O")
        self.assertEq
