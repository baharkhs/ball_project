# utils.py
# Utility functions for the ball simulation

# ball_simulation/utils.py

import random
import math

def calculate_distance(point1, point2):
    # Example function to calculate distance between two points
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

def generate_random_mass():
    # Generate a random mass, e.g., between 1 and 10
    return random.uniform(1, 10)

